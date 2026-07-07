#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import gc
import struct
import faulthandler

import numpy as np
import cv2
from pynq import allocate
from pynq_dpu import DpuOverlay

faulthandler.enable()

sys.path.append('/usr/lib/python3/site-packages')
sys.path.insert(0, '/home/xilinx/jupyter_notebooks/soft/DPU-PYNQ')

import vart
import xir


# ============================================================
# 0. Paths
# ============================================================
WORK_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae'
PL_DIR = os.path.join(WORK_DIR, 'zcu111_1920x1080')
BIT_PATH = os.path.join(PL_DIR, 'dpu.bit')

CODEBOOK_PATH = os.path.join(WORK_DIR, 'codebook.npy')
DEC_XMODEL = os.path.join(WORK_DIR, 'xmodel/decoder_1920x1080.xmodel')

RES_DIR = os.path.join(PL_DIR, 'results_1920x1080')
IDX_DIR = os.path.join(RES_DIR, 'idx_bins')
ZQ_DIR = os.path.join(RES_DIR, 'zq_npy')

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(IDX_DIR, exist_ok=True)
os.makedirs(ZQ_DIR, exist_ok=True)


# ============================================================
# 1. Shape / scale
# ============================================================
TARGET_W, TARGET_H = 1920, 1080
LATENT_W, LATENT_H = 480, 270

num_vectors = LATENT_W * LATENT_H
dim = 64
num_code = 512

expected_dec_in = (1, LATENT_H, LATENT_W, dim)
expected_dec_out = (1, TARGET_H, TARGET_W, 3)

dec_in_scale = 0.03125
dec_out_scale = 0.0078125
dec_scale_inv = 1.0 / dec_in_scale

print("=" * 80, flush=True)
print("3.run.py: idx -> vq_dequant -> Decoder", flush=True)
print("=" * 80, flush=True)
print("expected_dec_in :", expected_dec_in, flush=True)
print("expected_dec_out:", expected_dec_out, flush=True)
print("num_vectors     :", num_vectors, flush=True)
print("IDX_DIR         :", IDX_DIR, flush=True)
print("ZQ_DIR          :", ZQ_DIR, flush=True)
print("=" * 80, flush=True)


# ============================================================
# 2. Helpers
# ============================================================
def set_u64(mmio, lo_off, hi_off, addr):
    mmio.write(lo_off, addr & 0xFFFFFFFF)
    mmio.write(hi_off, (addr >> 32) & 0xFFFFFFFF)


def write_float(mmio, off, value):
    mmio.write(off, struct.unpack('<I', struct.pack('<f', np.float32(value)))[0])


def read_float(mmio, off):
    raw = mmio.read(off)
    return struct.unpack('<f', struct.pack('<I', raw))[0]


def start_and_wait_old_style(mmio, timeout_s=60.0):
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
        raise RuntimeError("No DPU subgraph found")

    return graph, dpu_subgraphs[0]


def get_fixpoint_scale(tensors):
    scales = []

    for t in tensors:
        try:
            fp = t.get_attr("fix_point")
            scale = 2 ** (-fp)
            print(f"  {t.name}: fix_point={fp}, scale={scale}", flush=True)
        except Exception:
            scale = 1.0
            print(f"  {t.name}: no fix_point, scale=1.0", flush=True)

        scales.append(scale)

    return scales


def free_buf(buf, name):
    try:
        if buf is not None:
            buf.freebuffer()
            print(f"  freed {name}", flush=True)
    except Exception as e:
        print(f"  warning: failed to free {name}: {e}", flush=True)


# ============================================================
# 3. Load overlay
# ============================================================
print("Load overlay...", flush=True)
overlay = DpuOverlay(BIT_PATH)

print("Overlay IPs:", flush=True)
for k in overlay.ip_dict.keys():
    print(" ", k, flush=True)

if hasattr(overlay, "vq_dequant_1"):
    vq_dequant = overlay.vq_dequant_1
elif hasattr(overlay, "vq_dequant_0"):
    vq_dequant = overlay.vq_dequant_0
else:
    raise RuntimeError("Cannot find vq_dequant_1 or vq_dequant_0")

print("vq_dequant register_map:", flush=True)
print(vq_dequant.register_map, flush=True)


# ============================================================
# 4. idx files
# ============================================================
if not os.path.exists(IDX_DIR):
    raise RuntimeError(f"IDX_DIR does not exist: {IDX_DIR}")

idx_files = sorted([f for f in os.listdir(IDX_DIR) if f.endswith('.bin')])

if len(idx_files) == 0:
    raise RuntimeError(f"No idx_*.bin found in {IDX_DIR}")

print(f"Found {len(idx_files)} idx files", flush=True)


# ============================================================
# 5. Load codebook
# ============================================================
codebook_np = np.load(CODEBOOK_PATH).astype(np.float32)
assert codebook_np.shape == (num_code, dim), f"codebook shape mismatch: {codebook_np.shape}"


# ============================================================
# 6. Phase A: idx -> vq_dequant -> zq_*.npy
# ============================================================
deq_times = []
zq_paths = []

print("\n[Phase A] idx -> vq_dequant -> zq_npy", flush=True)

for img_id, f in enumerate(idx_files):
    print(f"\n[Dequant] img={img_id}, file={f}", flush=True)

    idx_path = os.path.join(IDX_DIR, f)

    idx_np = np.fromfile(idx_path, dtype=np.uint16)

    if idx_np.shape != (num_vectors,):
        raise ValueError(f"idx shape mismatch: {idx_np.shape}, expected=({num_vectors},)")

    idx_min = int(idx_np.min())
    idx_max = int(idx_np.max())

    print(f"  idx range=[{idx_min},{idx_max}]", flush=True)

    if idx_min < 0 or idx_max >= num_code:
        raise RuntimeError(f"Invalid idx range: [{idx_min},{idx_max}]")

    # 每张图临时分配 vq_dequant buffer
    vq_codebook_buf = allocate(shape=(num_code, dim), dtype=np.float32, cacheable=1)
    vq_idx_buf = allocate(shape=(num_vectors,), dtype=np.uint16, cacheable=1)
    vq_zq_buf = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)

    vq_codebook_buf[:] = codebook_np
    vq_codebook_buf.sync_to_device()

    vq_idx_buf[:] = idx_np
    vq_idx_buf.sync_to_device()

    vq_zq_buf[:] = 0
    vq_zq_buf.sync_to_device()

    set_u64(vq_dequant.mmio, 0x1C, 0x20, vq_codebook_buf.device_address)
    set_u64(vq_dequant.mmio, 0x10, 0x14, vq_idx_buf.device_address)
    set_u64(vq_dequant.mmio, 0x28, 0x2C, vq_zq_buf.device_address)

    write_float(vq_dequant.mmio, 0x34, dec_scale_inv)

    print("  vq_dequant regs:", flush=True)
    print("    dec_scale_inv =", read_float(vq_dequant.mmio, 0x34), flush=True)

    print("  Run vq_dequant...", flush=True)
    t0 = time.time()

    start_and_wait_old_style(vq_dequant.mmio, timeout_s=60.0)

    deq_ms = (time.time() - t0) * 1000
    deq_times.append(deq_ms)

    print(f"  vq_dequant done: {deq_ms:.2f} ms", flush=True)

    vq_zq_buf.sync_from_device()

    zq_np = np.array(vq_zq_buf.reshape(expected_dec_in), dtype=np.int8, copy=True)

    print("  zq range:", int(zq_np.min()), int(zq_np.max()), flush=True)

    zq_path = os.path.join(ZQ_DIR, f'zq_{img_id:04d}.npy')
    np.save(zq_path, zq_np)
    zq_paths.append(zq_path)

    print("  saved:", zq_path, flush=True)

    free_buf(vq_codebook_buf, "vq_codebook_buf")
    free_buf(vq_idx_buf, "vq_idx_buf")
    free_buf(vq_zq_buf, "vq_zq_buf")

    del vq_codebook_buf
    del vq_idx_buf
    del vq_zq_buf
    del idx_np
    del zq_np

    gc.collect()
    time.sleep(0.05)

print("\nPhase A finished.", flush=True)
print(f"  avg vq_dequant = {np.mean(deq_times):.2f} ms", flush=True)


# ============================================================
# 7. Create Decoder runner
# ============================================================
print("\nCreate decoder runner...", flush=True)

_, dec_subgraph = get_dpu_subgraph(DEC_XMODEL)
dec_runner = vart.Runner.create_runner(dec_subgraph, "run")

dec_in_tensors = dec_runner.get_input_tensors()
dec_out_tensors = dec_runner.get_output_tensors()

print("Decoder input :", tuple(dec_in_tensors[0].dims), flush=True)
print("Decoder output:", tuple(dec_out_tensors[0].dims), flush=True)

if tuple(dec_in_tensors[0].dims) != expected_dec_in:
    raise ValueError(f"Decoder input mismatch: {tuple(dec_in_tensors[0].dims)}")

if tuple(dec_out_tensors[0].dims) != expected_dec_out:
    raise ValueError(f"Decoder output mismatch: {tuple(dec_out_tensors[0].dims)}")

print("Decoder fix_point:", flush=True)
dec_in_scales = get_fixpoint_scale(dec_in_tensors)
dec_out_scales = get_fixpoint_scale(dec_out_tensors)

dec_in_scale = dec_in_scales[0]
dec_out_scale = dec_out_scales[0]

print("Using dec_in_scale :", dec_in_scale, flush=True)
print("Using dec_out_scale:", dec_out_scale, flush=True)

# Decoder output LUT
post_lut = np.zeros(256, dtype=np.uint8)

for i in range(256):
    val_int8 = np.int8(i)
    val_fp32 = float(val_int8) * dec_out_scale
    val_norm = max(0.0, min(1.0, val_fp32 * 0.5 + 0.5))
    post_lut[i] = int(val_norm * 255.0)


# ============================================================
# 8. Phase B: zq -> Decoder -> recon
# ============================================================
dec_times = []

print("\n[Phase B] zq -> Decoder -> recon", flush=True)

for img_id, zq_path in enumerate(zq_paths):
    print(f"\n[Decoder] img={img_id}, zq={zq_path}", flush=True)

    zq_np = np.load(zq_path)

    if zq_np.shape != expected_dec_in:
        raise ValueError(f"zq shape mismatch: {zq_np.shape}, expected={expected_dec_in}")

    zq_np = np.ascontiguousarray(zq_np.astype(np.int8, copy=False))

    dec_out_cma = allocate(shape=expected_dec_out, dtype=np.int8, cacheable=0)
    dec_out_cma[:] = 0

    print(f"  dec_out_cma addr=0x{dec_out_cma.device_address:016X}, nbytes={dec_out_cma.nbytes}", flush=True)

    print("  Run Decoder...", flush=True)
    t0 = time.time()

    job_id = dec_runner.execute_async([zq_np], [dec_out_cma])
    dec_runner.wait(job_id)

    dec_ms = (time.time() - t0) * 1000
    dec_times.append(dec_ms)

    print(f"  Decoder done: {dec_ms:.2f} ms", flush=True)

    recon_img = post_lut[dec_out_cma[0].view(np.uint8)]

    save_path = os.path.join(RES_DIR, f'recon_{img_id}.png')
    cv2.imwrite(save_path, cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR))

    print("  saved:", save_path, flush=True)

    free_buf(dec_out_cma, "dec_out_cma")

    del dec_out_cma
    del zq_np
    del recon_img

    gc.collect()
    time.sleep(0.05)

print("\n3.run.py finished.", flush=True)
print(f"  avg vq_dequant = {np.mean(deq_times):.2f} ms", flush=True)
print(f"  avg Decoder    = {np.mean(dec_times):.2f} ms", flush=True)
print("  recon saved to :", RES_DIR, flush=True)

del dec_runner
gc.collect()

print("Decoder runner released.", flush=True)