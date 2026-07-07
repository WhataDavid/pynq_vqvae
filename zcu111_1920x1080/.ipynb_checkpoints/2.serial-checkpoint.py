#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import gc
import struct
import faulthandler

import numpy as np
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

PRE_DIR = os.path.join(WORK_DIR, 'imgs_preprocessed_1920x1080')
CODEBOOK_PATH = os.path.join(WORK_DIR, 'codebook.npy')
ENC_XMODEL = os.path.join(WORK_DIR, 'xmodel/encoder_1920x1080.xmodel')

RES_DIR = os.path.join(PL_DIR, 'results_1920x1080')
IDX_DIR = os.path.join(RES_DIR, 'idx_bins')

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(IDX_DIR, exist_ok=True)


# ============================================================
# 1. Shape / scale
# ============================================================
TARGET_W, TARGET_H = 1920, 1080
LATENT_W, LATENT_H = 480, 270

num_vectors = LATENT_W * LATENT_H      # 129600
dim = 64
num_code = 512

expected_enc_in = (1, TARGET_H, TARGET_W, 3)
expected_enc_out = (1, LATENT_H, LATENT_W, dim)

enc_in_scale = 0.015625
enc_out_scale = 0.015625

dec_in_scale = 0.03125
dec_scale_inv = 1.0 / dec_in_scale

print("=" * 80, flush=True)
print("2.run.py: Encoder(numpy buffer) -> vq_accel -> save idx", flush=True)
print("=" * 80, flush=True)
print("expected_enc_in :", expected_enc_in, flush=True)
print("expected_enc_out:", expected_enc_out, flush=True)
print("num_vectors     :", num_vectors, flush=True)
print("IDX_DIR         :", IDX_DIR, flush=True)
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

if hasattr(overlay, "vq_accel_1"):
    vq_accel = overlay.vq_accel_1
elif hasattr(overlay, "vq_accel_0"):
    vq_accel = overlay.vq_accel_0
else:
    raise RuntimeError("Cannot find vq_accel_1 or vq_accel_0")

print("vq_accel register_map:", flush=True)
print(vq_accel.register_map, flush=True)


# ============================================================
# 4. Create Encoder runner
# ============================================================
print("Create encoder runner...", flush=True)

_, enc_subgraph = get_dpu_subgraph(ENC_XMODEL)
enc_runner = vart.Runner.create_runner(enc_subgraph, "run")

enc_in_tensors = enc_runner.get_input_tensors()
enc_out_tensors = enc_runner.get_output_tensors()

print("Encoder input :", tuple(enc_in_tensors[0].dims), flush=True)
print("Encoder output:", tuple(enc_out_tensors[0].dims), flush=True)

if tuple(enc_in_tensors[0].dims) != expected_enc_in:
    raise ValueError(f"Encoder input mismatch: {tuple(enc_in_tensors[0].dims)}")

if tuple(enc_out_tensors[0].dims) != expected_enc_out:
    raise ValueError(f"Encoder output mismatch: {tuple(enc_out_tensors[0].dims)}")

print("Encoder fix_point:", flush=True)
enc_in_scales = get_fixpoint_scale(enc_in_tensors)
enc_out_scales = get_fixpoint_scale(enc_out_tensors)

enc_in_scale = enc_in_scales[0]
enc_out_scale = enc_out_scales[0]

print("Using enc_in_scale :", enc_in_scale, flush=True)
print("Using enc_out_scale:", enc_out_scale, flush=True)


# ============================================================
# 5. Data files
# ============================================================
if not os.path.exists(PRE_DIR):
    raise RuntimeError(f"PRE_DIR does not exist: {PRE_DIR}")

data_files = sorted([f for f in os.listdir(PRE_DIR) if f.endswith('.npy')])

if len(data_files) == 0:
    raise RuntimeError(f"No npy files found in {PRE_DIR}")

print(f"Found {len(data_files)} images in {PRE_DIR}", flush=True)


# ============================================================
# 6. Load codebook as normal numpy
#    注意：这里只是普通 numpy，不提前 allocate CMA
# ============================================================
codebook_np = np.load(CODEBOOK_PATH).astype(np.float32)
assert codebook_np.shape == (num_code, dim), f"codebook shape mismatch: {codebook_np.shape}"


# ============================================================
# 7. Run Encoder + vq_accel
# ============================================================
t_all = time.time()

enc_times = []
vq_alloc_times = []
vq_times = []
idx_save_times = []

for img_id, f in enumerate(data_files):
    print(f"\n[2.run] img={img_id}, file={f}", flush=True)

    # ------------------------------
    # 读入数据
    # 不计入 Encoder latency
    # ------------------------------
    path = os.path.join(PRE_DIR, f)
    data = np.load(path)

    print("  loaded:", data.shape, data.dtype, flush=True)

    if data.shape == (TARGET_H, TARGET_W, 3):
        pass
    elif data.shape == (1, TARGET_H, TARGET_W, 3):
        data = data[0]
    else:
        raise ValueError(f"Input shape mismatch: {data.shape}")

    if data.dtype == np.int8:
        enc_input_np = np.ascontiguousarray(data[np.newaxis])
    else:
        enc_input_np = np.clip(
            np.round(data.astype(np.float32) / enc_in_scale),
            -128,
            127
        ).astype(np.int8)[np.newaxis]
        enc_input_np = np.ascontiguousarray(enc_input_np)

    if enc_input_np.shape != expected_enc_in:
        raise ValueError(f"enc_input_np shape mismatch: {enc_input_np.shape}")

    print("  input range:", int(enc_input_np.min()), int(enc_input_np.max()), flush=True)

    # ------------------------------
    # Encoder 输出用普通 numpy
    # 这是你 benchmark 约 110 ms 的方式
    # ------------------------------
    enc_out_np = np.empty(expected_enc_out, dtype=np.int8, order='C')

    print("  Run Encoder with numpy input/output...", flush=True)
    t0 = time.perf_counter()

    job_id = enc_runner.execute_async([enc_input_np], [enc_out_np])
    enc_runner.wait(job_id)

    enc_ms = (time.perf_counter() - t0) * 1000
    enc_times.append(enc_ms)

    print(f"  Encoder done: {enc_ms:.2f} ms", flush=True)
    print("  enc_output range:", int(enc_out_np.min()), int(enc_out_np.max()), flush=True)

    # ------------------------------
    # Encoder 完成后再分配 vq_accel buffer
    # 不要提前分配，否则可能影响 Encoder 稳定性
    # ------------------------------
    print("  Allocate vq_accel buffers...", flush=True)
    t0 = time.perf_counter()

    vq_codebook_buf = allocate(shape=(num_code, dim), dtype=np.float32, cacheable=1)
    vq_in_buf = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)
    vq_idx_buf = allocate(shape=(num_vectors,), dtype=np.uint16, cacheable=1)

    vq_alloc_ms = (time.perf_counter() - t0) * 1000
    vq_alloc_times.append(vq_alloc_ms)

    print(f"  vq buffer alloc time: {vq_alloc_ms:.2f} ms", flush=True)
    print(f"    codebook addr=0x{vq_codebook_buf.device_address:016X}", flush=True)
    print(f"    vq_in    addr=0x{vq_in_buf.device_address:016X}", flush=True)
    print(f"    vq_idx   addr=0x{vq_idx_buf.device_address:016X}", flush=True)

    # ------------------------------
    # Copy to CMA and sync
    # 不计入 Encoder latency
    # ------------------------------
    vq_codebook_buf[:] = codebook_np
    vq_codebook_buf.sync_to_device()

    vq_in_buf[:] = enc_out_np.reshape(num_vectors, dim)
    vq_in_buf.sync_to_device()

    vq_idx_buf[:] = 0
    vq_idx_buf.sync_to_device()

    set_u64(vq_accel.mmio, 0x1C, 0x20, vq_codebook_buf.device_address)
    set_u64(vq_accel.mmio, 0x10, 0x14, vq_in_buf.device_address)
    set_u64(vq_accel.mmio, 0x28, 0x2C, vq_idx_buf.device_address)

    write_float(vq_accel.mmio, 0x34, enc_out_scale)
    write_float(vq_accel.mmio, 0x3C, dec_scale_inv)

    print("  vq_accel regs:", flush=True)
    print("    enc_scale     =", read_float(vq_accel.mmio, 0x34), flush=True)
    print("    dec_scale_inv =", read_float(vq_accel.mmio, 0x3C), flush=True)

    # ------------------------------
    # vq_accel latency
    # ------------------------------
    print("  Run vq_accel...", flush=True)
    t0 = time.perf_counter()

    start_and_wait_old_style(vq_accel.mmio, timeout_s=60.0)

    vq_ms = (time.perf_counter() - t0) * 1000
    vq_times.append(vq_ms)

    print(f"  vq_accel done: {vq_ms:.2f} ms", flush=True)

    vq_idx_buf.sync_from_device()

    idx_np = np.array(vq_idx_buf, dtype=np.uint16, copy=True)

    idx_min = int(idx_np.min())
    idx_max = int(idx_np.max())

    print(f"  idx range=[{idx_min},{idx_max}]", flush=True)

    if idx_min < 0 or idx_max >= num_code:
        raise RuntimeError(f"Invalid VQ index range: [{idx_min},{idx_max}]")

    # ------------------------------
    # 保存 idx
    # 不计入 Encoder latency
    # ------------------------------
    idx_path = os.path.join(IDX_DIR, f'idx_{img_id:04d}.bin')

    t0 = time.perf_counter()
    with open(idx_path, 'wb') as fp:
        fp.write(idx_np.tobytes())
    idx_save_ms = (time.perf_counter() - t0) * 1000
    idx_save_times.append(idx_save_ms)

    print(f"  saved: {idx_path}, save_time={idx_save_ms:.2f} ms", flush=True)

    # ------------------------------
    # 释放 vq 相关 buffer
    # ------------------------------
    free_buf(vq_codebook_buf, "vq_codebook_buf")
    free_buf(vq_in_buf, "vq_in_buf")
    free_buf(vq_idx_buf, "vq_idx_buf")

    del vq_codebook_buf
    del vq_in_buf
    del vq_idx_buf
    del idx_np
    del enc_out_np
    del enc_input_np
    del data

    gc.collect()
    time.sleep(0.02)


# ============================================================
# 8. Cleanup / summary
# ============================================================
total_ms = (time.time() - t_all) * 1000

print("\n2.run.py finished.", flush=True)
print(f"  avg Encoder numpy = {np.mean(enc_times):.2f} ms", flush=True)
print(f"  avg vq_alloc      = {np.mean(vq_alloc_times):.2f} ms", flush=True)
print(f"  avg vq_accel      = {np.mean(vq_times):.2f} ms", flush=True)
print(f"  avg idx_save      = {np.mean(idx_save_times):.2f} ms", flush=True)
print(f"  total wall time   = {total_ms:.2f} ms", flush=True)
print("  IDX saved to      :", IDX_DIR, flush=True)

del enc_runner
gc.collect()

print("Encoder runner released.", flush=True)