#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import struct
import numpy as np
import cv2
from pynq import allocate

# -----------------------------
# 配置
# -----------------------------
WORK_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae'
PRE_DIR  = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae/imgs_preprocessed'
RES_DIR  = os.path.join(WORK_DIR, 'results')
os.makedirs(RES_DIR, exist_ok=True)

BIT_PATH       = os.path.join(WORK_DIR, 'pl_vq_parallel_260513/dpu.bit')
ENC_XMODEL     = os.path.join(WORK_DIR, 'xmodel/encoder_768x512.xmodel')
DEC_XMODEL     = os.path.join(WORK_DIR, 'xmodel/decoder_768x512.xmodel')
CODEBOOK_PATH  = os.path.join(WORK_DIR, 'codebook.npy')

ENC_OUT_SCALE  = 0.015625
DEC_IN_SCALE   = 0.03125
DEC_OUT_SCALE  = 0.007812

# VQ 控制寄存器（你之前已验证可用）
REG_AP_CTRL      = 0x00
REG_IN_ADDR_L    = 0x10
REG_IN_ADDR_H    = 0x14
REG_OUT_ADDR_L   = 0x28
REG_OUT_ADDR_H   = 0x2C

# 可选参数寄存器（如果你的 HLS 顶层有这些端口，就启用；否则保持 False）
USE_RUNTIME_SHAPE_REG = False
REG_NUM_VECTORS = 0x30
REG_DIM         = 0x34

# -----------------------------
# 环境导入
# -----------------------------
sys.path.append('/usr/lib/python3/site-packages')
sys.path.insert(0, '/home/xilinx/jupyter_notebooks/soft/DPU-PYNQ')
from pynq_dpu import DpuOverlay
import vart
import xir

def f32_to_u32(v):
    return struct.unpack('<I', struct.pack('<f', float(v)))[0]

def get_subgraph(xmodel_path):
    g = xir.Graph.deserialize(xmodel_path)
    return g.get_root_subgraph().toposort_child_subgraph()[1]

def to_hwc(arr):
    a = arr[0] if arr.ndim == 4 else arr
    if a.shape[-1] in (1, 3):
        return a
    if a.shape[0] in (1, 3):
        return np.transpose(a, (1, 2, 0))
    raise RuntimeError(f"Bad tensor shape: {arr.shape}")

def postprocess_int8_to_u8(dec_out_int8):
    img = to_hwc(dec_out_int8).astype(np.float32)
    img = img * DEC_OUT_SCALE
    img = np.clip(img * 0.5 + 0.5, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)

def save_rgb(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

def run():
    print("⏳ 连接 overlay...")
    overlay = DpuOverlay(BIT_PATH, download=False)
    vq_ip = overlay.vq_accel_1
    print("✅ overlay connected")

    enc_runner = vart.Runner.create_runner(get_subgraph(ENC_XMODEL), "run")
    dec_runner = vart.Runner.create_runner(get_subgraph(DEC_XMODEL), "run")

    enc_in_shape  = tuple(enc_runner.get_input_tensors()[0].dims)
    enc_out_shape = tuple(enc_runner.get_output_tensors()[0].dims)
    dec_in_shape  = tuple(dec_runner.get_input_tensors()[0].dims)
    dec_out_shape = tuple(dec_runner.get_output_tensors()[0].dims)

    print("ENC input :", enc_in_shape)
    print("ENC output:", enc_out_shape)
    print("DEC input :", dec_in_shape)
    print("DEC output:", dec_out_shape)

    assert enc_out_shape == dec_in_shape, "Encoder output / Decoder input mismatch"
    assert len(enc_out_shape) == 4, f"Unexpected latent shape: {enc_out_shape}"

    # 按你当前 xmodel：NHWC
    _, h_lat, w_lat, c_lat = enc_out_shape
    num_vectors = h_lat * w_lat
    dim = c_lat
    print(f"latent: H={h_lat}, W={w_lat}, C={c_lat}, num_vectors={num_vectors}")

    # 分配 VQ buffer
    vq_in  = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)
    vq_out = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)

    # codebook
    codebook = np.load(CODEBOOK_PATH).astype(np.float32)
    assert codebook.shape[1] == dim, f"codebook dim mismatch: {codebook.shape}, dim={dim}"

    cb_buf = allocate(shape=codebook.shape, dtype=np.float32, cacheable=1)
    cb_buf[:] = codebook
    cb_buf.sync_to_device()

    # 仅写已知存在字段（避免 register_map 乱探测导致崩溃）
    vq_ip.register_map.in_codebook_1 = cb_buf.device_address & 0xFFFFFFFF
    vq_ip.register_map.in_codebook_2 = cb_buf.device_address >> 32
    vq_ip.register_map.enc_scale     = f32_to_u32(ENC_OUT_SCALE)
    vq_ip.register_map.dec_scale_inv = f32_to_u32(1.0 / DEC_IN_SCALE)

    files = sorted([f for f in os.listdir(PRE_DIR) if f.endswith('.npy')])
    print(f"📦 images: {len(files)}")

    for i, fn in enumerate(files):
        x = np.load(os.path.join(PRE_DIR, fn))  # HWC int8, shape (512,768,3)

        # 1) Encoder
        enc_out = np.empty(enc_out_shape, dtype=np.int8, order='C')
        jid = enc_runner.execute_async([np.ascontiguousarray(x[np.newaxis], dtype=np.int8)], [enc_out])
        enc_runner.wait(jid)

        # 2) VQ
        src_flat = enc_out.reshape(num_vectors, dim)
        np.copyto(vq_in, src_flat)
        vq_out.fill(0)

        vq_in.sync_to_device()

        vq_ip.mmio.write(REG_IN_ADDR_L,  vq_in.device_address & 0xFFFFFFFF)
        vq_ip.mmio.write(REG_IN_ADDR_H,  vq_in.device_address >> 32)
        vq_ip.mmio.write(REG_OUT_ADDR_L, vq_out.device_address & 0xFFFFFFFF)
        vq_ip.mmio.write(REG_OUT_ADDR_H, vq_out.device_address >> 32)

        if USE_RUNTIME_SHAPE_REG:
            vq_ip.mmio.write(REG_NUM_VECTORS, int(num_vectors))
            vq_ip.mmio.write(REG_DIM, int(dim))

        vq_ip.mmio.write(REG_AP_CTRL, 0x11)
        while (vq_ip.mmio.read(REG_AP_CTRL) & 0x02) == 0:
            time.sleep(0.0005)

        vq_out.sync_from_device()
        zq = np.array(vq_out, copy=True).reshape(dec_in_shape)

        # 3) Decoder
        dec_out = np.empty(dec_out_shape, dtype=np.int8, order='C')
        jid = dec_runner.execute_async([np.ascontiguousarray(zq, dtype=np.int8)], [dec_out])
        dec_runner.wait(jid)

        # 4) 保存
        img = postprocess_int8_to_u8(dec_out)
        save_rgb(os.path.join(RES_DIR, f"recon_{i}.png"), img)

        if i < 3:
            arr = np.array(vq_out, copy=False)
            print(f"[VQ stat {i}] min={arr.min()} max={arr.max()} mean={arr.mean():.3f} nz={np.count_nonzero(arr)}")

    del enc_runner
    del dec_runner
    print("✅ done:", RES_DIR)

if __name__ == "__main__":
    run()