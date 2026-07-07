#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
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

RES_DIR = './results_768x512_serial'
IDX_DIR = os.path.join(RES_DIR, 'idx_bins')

sys.path.append('/usr/lib/python3/site-packages')
sys.path.insert(0, '/home/xilinx/jupyter_notebooks/soft/DPU-PYNQ')

from pynq_dpu import DpuOverlay
import vart
import xir

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(IDX_DIR, exist_ok=True)


# ============================================================
# Parameters
# ============================================================
enc_out_scale = 0.015625
dec_in_scale = 0.03125
dec_out_scale = 0.007812
dec_scale_inv = 1.0 / dec_in_scale

IMG_H = 512
IMG_W = 768
LATENT_H = 128
LATENT_W = 192

num_vectors = LATENT_H * LATENT_W
dim = 64
num_code = 512

expected_enc_in = (1, IMG_H, IMG_W, 3)
expected_enc_out = (1, LATENT_H, LATENT_W, dim)
expected_dec_in = (1, LATENT_H, LATENT_W, dim)
expected_dec_out = (1, IMG_H, IMG_W, 3)


# ============================================================
# Helper functions
# ============================================================
def set_u64(mmio, lo_off, hi_off, addr):
    mmio.write(lo_off, addr & 0xFFFFFFFF)
    mmio.write(hi_off, (addr >> 32) & 0xFFFFFFFF)


def write_float(mmio, off, value):
    mmio.write(off, struct.unpack('<I', struct.pack('<f', np.float32(value)))[0])


def read_float(mmio, off):
    raw = mmio.read(off)
    return struct.unpack('<f', struct.pack('<I', raw))[0]


def start_and_wait_old_style(mmio, timeout_s=5.0):
    mmio.write(0x00, 0x11)

    t0 = time.time()
    while (mmio.read(0x00) & 0x02) == 0:
        if time.time() - t0 > timeout_s:
            ctrl = mmio.read(0x00)
            raise RuntimeError(f"IP timeout waiting for AP_DONE, CTRL=0x{ctrl:08X}")
        time.sleep(0.0001)


def get_dpu_subgraph(path):
    graph = xir.Graph.deserialize(path)
    root = graph.get_root_subgraph()
    children = root.toposort_child_subgraph()

    for i, s in enumerate(children):
        dev = s.get_attr("device") if s.has_attr("device") else "NONE"
        print(f"[{i}] {s.get_name()} device={dev}", flush=True)

    dpu_subgraphs = [
        s for s in children
        if s.has_attr("device") and s.get_attr("device").upper() == "DPU"
    ]

    if len(dpu_subgraphs) == 0:
        raise RuntimeError(f"No DPU subgraph found in {path}")

    return graph, dpu_subgraphs[0]


def free_buf(buf, name):
    try:
        if buf is not None:
            buf.freebuffer()
            print(f"  freed {name}", flush=True)
    except Exception as e:
        print(f"  warning: failed to free {name}: {e}", flush=True)


# ============================================================
# Load overlay and runners
# ============================================================
print("=" * 80, flush=True)
print("768x512 SERIAL Pipeline", flush=True)
print("Read -> Encoder -> vq_accel -> vq_dequant -> Decoder -> LUT -> Save", flush=True)
print("=" * 80, flush=True)

print("Attaching to already-loaded bitstream...", flush=True)
overlay = DpuOverlay(BIT_PATH, download=False)

vq_accel = overlay.vq_accel_1
vq_dequant = overlay.vq_dequant_1

print("Overlay IPs:", flush=True)
for k in overlay.ip_dict.keys():
    print(" ", k, flush=True)

print("\nCreate encoder runner...", flush=True)
_, enc_subgraph = get_dpu_subgraph(ENC_XMODEL)
enc_runner = vart.Runner.create_runner(enc_subgraph, "run")

print("\nCreate decoder runner...", flush=True)
_, dec_subgraph = get_dpu_subgraph(DEC_XMODEL)
dec_runner = vart.Runner.create_runner(dec_subgraph, "run")

print("Encoder input :", tuple(enc_runner.get_input_tensors()[0].dims), flush=True)
print("Encoder output:", tuple(enc_runner.get_output_tensors()[0].dims), flush=True)
print("Decoder input :", tuple(dec_runner.get_input_tensors()[0].dims), flush=True)
print("Decoder output:", tuple(dec_runner.get_output_tensors()[0].dims), flush=True)

if tuple(enc_runner.get_input_tensors()[0].dims) != expected_enc_in:
    raise ValueError("Encoder input shape mismatch")

if tuple(enc_runner.get_output_tensors()[0].dims) != expected_enc_out:
    raise ValueError("Encoder output shape mismatch")

if tuple(dec_runner.get_input_tensors()[0].dims) != expected_dec_in:
    raise ValueError("Decoder input shape mismatch")

if tuple(dec_runner.get_output_tensors()[0].dims) != expected_dec_out:
    raise ValueError("Decoder output shape mismatch")


# ============================================================
# Allocate PL buffers
# ============================================================
print("\nAllocate PL buffers...", flush=True)

vq1_stage_in_buf = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)
vq1_stage_idx_buf = allocate(shape=(num_vectors,), dtype=np.uint16, cacheable=1)

vq2_stage_idx_buf = allocate(shape=(num_vectors,), dtype=np.uint16, cacheable=1)
vq2_stage_zq_buf = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)

codebook = np.load(CODEBOOK_PATH).astype(np.float32)
assert codebook.shape == (num_code, dim), f"codebook shape mismatch: {codebook.shape}"

vq_codebook_buf = allocate(shape=(num_code, dim), dtype=np.float32, cacheable=1)
vq_codebook_buf[:] = codebook
vq_codebook_buf.sync_to_device()

print(f"vq_codebook addr = 0x{vq_codebook_buf.device_address:016X}", flush=True)
print(f"vq1_in      addr = 0x{vq1_stage_in_buf.device_address:016X}", flush=True)
print(f"vq1_idx     addr = 0x{vq1_stage_idx_buf.device_address:016X}", flush=True)
print(f"vq2_idx     addr = 0x{vq2_stage_idx_buf.device_address:016X}", flush=True)
print(f"vq2_zq      addr = 0x{vq2_stage_zq_buf.device_address:016X}", flush=True)

# Configure static registers
set_u64(vq_accel.mmio, 0x1C, 0x20, vq_codebook_buf.device_address)
write_float(vq_accel.mmio, 0x34, enc_out_scale)
write_float(vq_accel.mmio, 0x3C, dec_scale_inv)

set_u64(vq_dequant.mmio, 0x1C, 0x20, vq_codebook_buf.device_address)
write_float(vq_dequant.mmio, 0x34, dec_scale_inv)

print("\nScale regs:", flush=True)
print("  vq_accel.enc_scale       =", read_float(vq_accel.mmio, 0x34), flush=True)
print("  vq_accel.dec_scale_inv   =", read_float(vq_accel.mmio, 0x3C), flush=True)
print("  vq_dequant.dec_scale_inv =", read_float(vq_dequant.mmio, 0x34), flush=True)


# ============================================================
# Data files and LUT
# ============================================================
data_files = sorted([f for f in os.listdir(PRE_DIR) if f.endswith('.npy')])
num_imgs = len(data_files)

if num_imgs == 0:
    raise RuntimeError(f"No .npy files found in {PRE_DIR}")

print(f"\nFound {num_imgs} images in {PRE_DIR}", flush=True)

post_lut = np.zeros(256, dtype=np.uint8)

for i in range(256):
    val_int8 = np.int8(i)
    val_fp32 = float(val_int8) * dec_out_scale
    val_norm = max(0.0, min(1.0, val_fp32 * 0.5 + 0.5))
    post_lut[i] = int(val_norm * 255.0)


# ============================================================
# Serial pipeline
# ============================================================
enc_times = []
vq_times = []
deq_times = []
dec_times = []
lut_save_times = []

print("\nStart serial pipeline...", flush=True)
t_all = time.time()

for img_id, filename in enumerate(data_files):
    print(f"\n[{img_id}/{num_imgs}] file={filename}", flush=True)

    # --------------------------------------------------------
    # 1. Read input
    # --------------------------------------------------------
    input_path = os.path.join(PRE_DIR, filename)
    input_data = np.load(input_path)

    print("  input:", input_data.shape, input_data.dtype, flush=True)

    if input_data.shape == (IMG_H, IMG_W, 3):
        enc_input_np = np.ascontiguousarray(input_data[np.newaxis])
    elif input_data.shape == expected_enc_in:
        enc_input_np = np.ascontiguousarray(input_data)
    else:
        raise ValueError(f"Input shape mismatch: {input_data.shape}")

    if enc_input_np.dtype != np.int8:
        raise ValueError(f"Expected int8 preprocessed input, got {enc_input_np.dtype}")

    # --------------------------------------------------------
    # 2. Encoder
    # --------------------------------------------------------
    enc_out_np = np.empty(expected_enc_out, dtype=np.int8, order='C')

    print("  Run Encoder...", flush=True)
    t0 = time.perf_counter()

    job_id = enc_runner.execute_async([enc_input_np], [enc_out_np])
    enc_runner.wait(job_id)

    enc_ms = (time.perf_counter() - t0) * 1000
    enc_times.append(enc_ms)

    print(f"  Encoder done: {enc_ms:.2f} ms", flush=True)
    print(f"  enc_out range: [{int(enc_out_np.min())}, {int(enc_out_np.max())}]", flush=True)

    # --------------------------------------------------------
    # 3. vq_accel
    # --------------------------------------------------------
    vq1_stage_in_buf[:] = enc_out_np.reshape(num_vectors, dim)
    vq1_stage_in_buf.sync_to_device()

    vq1_stage_idx_buf[:] = 0
    vq1_stage_idx_buf.sync_to_device()

    set_u64(vq_accel.mmio, 0x10, 0x14, vq1_stage_in_buf.device_address)
    set_u64(vq_accel.mmio, 0x28, 0x2C, vq1_stage_idx_buf.device_address)

    print("  Run vq_accel...", flush=True)
    t0 = time.perf_counter()

    start_and_wait_old_style(vq_accel.mmio, timeout_s=5.0)

    vq_ms = (time.perf_counter() - t0) * 1000
    vq_times.append(vq_ms)

    vq1_stage_idx_buf.sync_from_device()
    idx_snapshot = np.array(vq1_stage_idx_buf, dtype=np.uint16, copy=True)

    idx_min = int(idx_snapshot.min())
    idx_max = int(idx_snapshot.max())

    print(f"  vq_accel done: {vq_ms:.2f} ms", flush=True)
    print(f"  idx range: [{idx_min}, {idx_max}]", flush=True)

    if idx_min < 0 or idx_max >= num_code:
        raise RuntimeError(f"Invalid index range: [{idx_min}, {idx_max}]")

    idx_path = os.path.join(IDX_DIR, f'idx_{img_id:04d}.bin')
    with open(idx_path, 'wb') as fp:
        fp.write(idx_snapshot.tobytes())

    # --------------------------------------------------------
    # 4. vq_dequant
    # --------------------------------------------------------
    vq2_stage_idx_buf[:] = idx_snapshot
    vq2_stage_idx_buf.sync_to_device()

    vq2_stage_zq_buf[:] = 0
    vq2_stage_zq_buf.sync_to_device()

    set_u64(vq_dequant.mmio, 0x10, 0x14, vq2_stage_idx_buf.device_address)
    set_u64(vq_dequant.mmio, 0x28, 0x2C, vq2_stage_zq_buf.device_address)

    print("  Run vq_dequant...", flush=True)
    t0 = time.perf_counter()

    start_and_wait_old_style(vq_dequant.mmio, timeout_s=5.0)

    deq_ms = (time.perf_counter() - t0) * 1000
    deq_times.append(deq_ms)

    vq2_stage_zq_buf.sync_from_device()
    zq_snapshot = np.array(vq2_stage_zq_buf, dtype=np.int8, copy=True)

    print(f"  vq_dequant done: {deq_ms:.2f} ms", flush=True)
    print(f"  zq range: [{int(zq_snapshot.min())}, {int(zq_snapshot.max())}]", flush=True)

    # --------------------------------------------------------
    # 5. Decoder
    # --------------------------------------------------------
    dec_in_np = np.ascontiguousarray(zq_snapshot.reshape(expected_dec_in))
    dec_out_np = np.empty(expected_dec_out, dtype=np.int8, order='C')

    print("  Run Decoder...", flush=True)
    t0 = time.perf_counter()

    job_id = dec_runner.execute_async([dec_in_np], [dec_out_np])
    dec_runner.wait(job_id)

    dec_ms = (time.perf_counter() - t0) * 1000
    dec_times.append(dec_ms)

    print(f"  Decoder done: {dec_ms:.2f} ms", flush=True)

    # --------------------------------------------------------
    # 6. LUT + save image
    # --------------------------------------------------------
    t0 = time.perf_counter()

    recon_img = post_lut[dec_out_np[0].view(np.uint8)]

    save_path = os.path.join(RES_DIR, f'recon_{img_id}.png')
    cv2.imwrite(save_path, cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR))

    lut_save_ms = (time.perf_counter() - t0) * 1000
    lut_save_times.append(lut_save_ms)

    print(f"  saved: {save_path}, LUT+save={lut_save_ms:.2f} ms", flush=True)

    del input_data
    del enc_input_np
    del enc_out_np
    del idx_snapshot
    del zq_snapshot
    del dec_in_np
    del dec_out_np
    del recon_img


# ============================================================
# Summary
# ============================================================
total_time = time.time() - t_all

print("\n" + "=" * 80, flush=True)
print("Serial pipeline finished.", flush=True)
print("=" * 80, flush=True)
print(f"Images        : {num_imgs}", flush=True)
print(f"Total time    : {total_time * 1000:.2f} ms", flush=True)
print(f"FPS           : {num_imgs / total_time:.2f}", flush=True)
print(f"avg Encoder   : {np.mean(enc_times):.2f} ms", flush=True)
print(f"avg vq_accel  : {np.mean(vq_times):.2f} ms", flush=True)
print(f"avg vq_dequant: {np.mean(deq_times):.2f} ms", flush=True)
print(f"avg Decoder   : {np.mean(dec_times):.2f} ms", flush=True)
print(f"avg LUT+save  : {np.mean(lut_save_times):.2f} ms", flush=True)
print(f"Saved images  : {RES_DIR}", flush=True)
print(f"Saved idx bins: {IDX_DIR}", flush=True)


# ============================================================
# Cleanup
# ============================================================
free_buf(vq1_stage_in_buf, "vq1_stage_in_buf")
free_buf(vq1_stage_idx_buf, "vq1_stage_idx_buf")
free_buf(vq2_stage_idx_buf, "vq2_stage_idx_buf")
free_buf(vq2_stage_zq_buf, "vq2_stage_zq_buf")
free_buf(vq_codebook_buf, "vq_codebook_buf")

del enc_runner
del dec_runner

print("DPU runners released.", flush=True)
print("Done.", flush=True)