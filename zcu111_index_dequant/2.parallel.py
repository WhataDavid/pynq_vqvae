#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import threading
import queue
import struct

import numpy as np
import cv2
from pynq import allocate

WORK_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae'
PL_DIR = os.path.join(WORK_DIR, 'zcu111_index_dequant')
PRE_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae/imgs_preprocessed'
CODEBOOK_PATH = os.path.join(WORK_DIR, 'codebook.npy')
ENC_XMODEL = os.path.join(WORK_DIR, 'xmodel/encoder_768x512.xmodel')
DEC_XMODEL = os.path.join(WORK_DIR, 'xmodel/decoder_768x512.xmodel')
BIT_PATH = os.path.join(PL_DIR, 'dpu_debug.bit')

RES_DIR = './results_768x512'
IDX_DIR = os.path.join(RES_DIR, 'idx_bins')

sys.path.append('/usr/lib/python3/site-packages')
sys.path.insert(0, '/home/xilinx/jupyter_notebooks/soft/DPU-PYNQ')

from pynq_dpu import DpuOverlay
import vart
import xir

if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)
if not os.path.exists(IDX_DIR):
    os.makedirs(IDX_DIR)

enc_out_scale = 0.015625
dec_in_scale = 0.03125
dec_out_scale = 0.007812
dec_scale_inv = 1.0 / dec_in_scale

num_vectors = 128 * 192
dim = 64
num_code = 512
num_bufs = 3


def set_u64(mmio, lo_off, hi_off, addr):
    mmio.write(lo_off, addr & 0xFFFFFFFF)
    mmio.write(hi_off, (addr >> 32) & 0xFFFFFFFF)


def write_float(mmio, off, value):
    mmio.write(off, struct.unpack('<I', struct.pack('<f', np.float32(value)))[0])


def start_and_wait_old_style(mmio, timeout_s=2.0):
    mmio.write(0x00, 0x11)
    t0 = time.time()
    while (mmio.read(0x00) & 0x02) == 0:
        if time.time() - t0 > timeout_s:
            raise RuntimeError("IP timeout waiting for AP_DONE")
        time.sleep(0.0001)


def get_dpu_subgraph(path):
    graph = xir.Graph.deserialize(path)
    return graph, graph.get_root_subgraph().toposort_child_subgraph()[1]


print("Attaching to already-loaded bitstream...", flush=True)
overlay = DpuOverlay(BIT_PATH, download=False)
vq_accel = overlay.vq_accel_1
vq_dequant = overlay.vq_dequant_1

_, enc_subgraph = get_dpu_subgraph(ENC_XMODEL)
enc_runner = vart.Runner.create_runner(enc_subgraph, "run")

_, dec_subgraph = get_dpu_subgraph(DEC_XMODEL)
dec_runner = vart.Runner.create_runner(dec_subgraph, "run")

enc_feat_bufs = [
    allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)
    for _ in range(num_bufs)
]

vq1_stage_in_buf = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)
vq1_stage_idx_buf = allocate(shape=(num_vectors,), dtype=np.uint16, cacheable=1)

vq2_stage_idx_buf = allocate(shape=(num_vectors,), dtype=np.uint16, cacheable=1)
vq2_stage_zq_buf = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)

codebook = np.load(CODEBOOK_PATH).astype(np.float32)
assert codebook.shape == (num_code, dim), "codebook shape mismatch"

vq_codebook_buf = allocate(shape=(num_code, dim), dtype=np.float32, cacheable=1)
vq_codebook_buf[:] = codebook
vq_codebook_buf.sync_to_device()

set_u64(vq_accel.mmio, 0x1C, 0x20, vq_codebook_buf.device_address)
write_float(vq_accel.mmio, 0x34, enc_out_scale)
write_float(vq_accel.mmio, 0x3C, dec_scale_inv)

set_u64(vq_dequant.mmio, 0x1C, 0x20, vq_codebook_buf.device_address)
write_float(vq_dequant.mmio, 0x34, dec_scale_inv)

data_files = sorted([f for f in os.listdir(PRE_DIR) if f.endswith('.npy')])
num_imgs = len(data_files)

post_lut = np.zeros(256, dtype=np.uint8)
for i in range(256):
    val_int8 = np.int8(i)
    val_fp32 = float(val_int8) * dec_out_scale
    val_norm = max(0.0, min(1.0, val_fp32 * 0.5 + 0.5))
    post_lut[i] = int(val_norm * 255.0)


def phase1_pipeline():
    read_queue = queue.Queue(maxsize=3)
    free_queue = queue.Queue(maxsize=num_bufs)
    enc_res_queue = queue.Queue(maxsize=num_bufs)

    for i in range(num_bufs):
        free_queue.put(i)

    all_idx_results = [None] * num_imgs

    def read_worker():
        for img_id, f in enumerate(data_files):
            data = np.load(os.path.join(PRE_DIR, f))
            read_queue.put((img_id, data))
        read_queue.put(None)

    def enc_worker():
        while True:
            item = read_queue.get()
            if item is None:
                enc_res_queue.put(None)
                break

            img_id, input_data = item
            in_buf = [np.ascontiguousarray(input_data[np.newaxis])]
            buf_idx = free_queue.get()
            target_feat_buf = enc_feat_bufs[buf_idx]

            out_buf = [np.ndarray((1, 128, 192, 64), dtype=np.int8, buffer=target_feat_buf.data)]
            job_id = enc_runner.execute_async(in_buf, out_buf)
            enc_runner.wait(job_id)

            enc_res_queue.put((img_id, buf_idx))

    print(f"[Phase 1] Read -> Encoder -> vq_accel: {num_imgs} images")
    start_time = time.time()

    t_read = threading.Thread(target=read_worker)
    t_enc = threading.Thread(target=enc_worker)
    t_read.start()
    t_enc.start()

    while True:
        item = enc_res_queue.get()
        if item is None:
            break

        img_id, buf_idx = item
        curr_feat_buf = enc_feat_bufs[buf_idx]
        curr_feat_buf.sync_from_device()

        vq1_stage_in_buf[:] = curr_feat_buf
        vq1_stage_in_buf.sync_to_device()

        vq1_stage_idx_buf[:] = 0
        vq1_stage_idx_buf.sync_to_device()

        set_u64(vq_accel.mmio, 0x10, 0x14, vq1_stage_in_buf.device_address)
        set_u64(vq_accel.mmio, 0x28, 0x2C, vq1_stage_idx_buf.device_address)
        start_and_wait_old_style(vq_accel.mmio)

        vq1_stage_idx_buf.sync_from_device()
        idx_snapshot = np.array(vq1_stage_idx_buf, copy=True)
        all_idx_results[img_id] = idx_snapshot

        idx_path = os.path.join(IDX_DIR, f'idx_{img_id:04d}.bin')
        with open(idx_path, 'wb') as f:
            f.write(idx_snapshot.tobytes())

        free_queue.put(buf_idx)

    t_read.join()
    t_enc.join()

    phase1_time = time.time() - start_time
    print(f"[Phase 1] Finished! Time: {phase1_time*1000:.2f} ms | FPS: {num_imgs/phase1_time:.2f}\n")
    return all_idx_results


def phase2_pipeline(all_idx_results):
    dec_out_queue = queue.Queue(maxsize=3)
    all_recon_imgs = [None] * num_imgs

    def dec_worker():
        for img_id, idx_res in enumerate(all_idx_results):
            if idx_res is None:
                dec_out_queue.put((img_id, None))
                continue

            vq2_stage_idx_buf[:] = idx_res
            vq2_stage_idx_buf.sync_to_device()

            vq2_stage_zq_buf[:] = 0
            vq2_stage_zq_buf.sync_to_device()

            set_u64(vq_dequant.mmio, 0x10, 0x14, vq2_stage_idx_buf.device_address)
            set_u64(vq_dequant.mmio, 0x28, 0x2C, vq2_stage_zq_buf.device_address)
            start_and_wait_old_style(vq_dequant.mmio)

            vq2_stage_zq_buf.sync_from_device()
            zq_snapshot = np.array(vq2_stage_zq_buf, copy=True)

            dec_in_buf = [zq_snapshot.reshape(1, 128, 192, 64)]
            dec_out_buf = [np.empty((1, 512, 768, 3), dtype=np.int8, order='C')]

            job_id = dec_runner.execute_async(dec_in_buf, dec_out_buf)
            dec_runner.wait(job_id)

            dec_out_queue.put((img_id, dec_out_buf[0].copy()))

        dec_out_queue.put(None)

    def lut_worker():
        while True:
            item = dec_out_queue.get()
            if item is None:
                break

            img_id, dec_data = item
            if dec_data is None:
                continue

            recon_img = post_lut[dec_data[0].view(np.uint8)]
            all_recon_imgs[img_id] = recon_img

    print(f"[Phase 2] vq_dequant -> Decoder -> LUT: {num_imgs} images")
    start_time = time.time()

    t_dec = threading.Thread(target=dec_worker)
    t_lut = threading.Thread(target=lut_worker)
    t_dec.start()
    t_lut.start()

    t_dec.join()
    t_lut.join()

    phase2_time = time.time() - start_time
    print(f"[Phase 2] Finished! Time: {phase2_time*1000:.2f} ms | FPS: {num_imgs/phase2_time:.2f}\n")
    return all_recon_imgs


def save_images(all_recon_imgs):
    print("Writing final images...")
    saved = 0
    for i, img in enumerate(all_recon_imgs):
        if img is None:
            continue
        cv2.imwrite(f'{RES_DIR}/recon_{i}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        saved += 1

    print(f"Pipeline completed. Saved {saved} images.")
    print(f"Saved idx bins to: {IDX_DIR}")


if __name__ == "__main__":
    all_idx_results = phase1_pipeline()
    all_recon_imgs = phase2_pipeline(all_idx_results)
    save_images(all_recon_imgs)

    del enc_runner
    del dec_runner
    print("DPU runners released.")