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

CODEBOOK_PATH = os.path.join(WORK_DIR, 'codebook_som.npy')
DEC_XMODEL = os.path.join(WORK_DIR, 'xmodel/decoder_1920x1080.xmodel')

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

expected_dec_in = (1, LATENT_H, LATENT_W, dim)
expected_dec_out = (1, TARGET_H, TARGET_W, 3)

dec_in_scale = 0.03125
dec_out_scale = 0.0078125
dec_scale_inv = 1.0 / dec_in_scale


# ============================================================
# 2. Runtime options
# ============================================================
VERBOSE = False
CHECK_RANGE = True

# 如果你 idx_bins 里有重复 benchmark 产生的 idx_0000 ~ idx_0049，
# 这里会全部处理。正式只想处理前 10 张可改为 10。
MAX_FILES = None

VQ_TIMEOUT_S = 60.0


print("=" * 80, flush=True)
print("3.run_no_zq_save.py: idx -> vq_dequant -> Decoder -> recon", flush=True)
print("No zq_npy save/load", flush=True)
print("=" * 80, flush=True)
print("expected_dec_in :", expected_dec_in, flush=True)
print("expected_dec_out:", expected_dec_out, flush=True)
print("num_vectors     :", num_vectors, flush=True)
print("IDX_DIR         :", IDX_DIR, flush=True)
print("RES_DIR         :", RES_DIR, flush=True)
print("VERBOSE         :", VERBOSE, flush=True)
print("CHECK_RANGE     :", CHECK_RANGE, flush=True)
print("=" * 80, flush=True)


# ============================================================
# 3. Helpers
# ============================================================
def log(msg):
    if VERBOSE:
        print(msg, flush=True)


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


def print_avg(name, arr):
    if len(arr) == 0:
        print(f"  avg {name:<18} = N/A", flush=True)
    else:
        print(f"  avg {name:<18} = {np.mean(arr):.2f} ms", flush=True)


# ============================================================
# 4. Load overlay
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
# 5. idx files
# ============================================================
if not os.path.exists(IDX_DIR):
    raise RuntimeError(f"IDX_DIR does not exist: {IDX_DIR}")

idx_files = sorted([f for f in os.listdir(IDX_DIR) if f.endswith('.bin')])

if len(idx_files) == 0:
    raise RuntimeError(f"No idx_*.bin found in {IDX_DIR}")

if MAX_FILES is not None:
    idx_files = idx_files[:MAX_FILES]

num_imgs = len(idx_files)

print(f"Found {num_imgs} idx files", flush=True)


# ============================================================
# 6. Load codebook
# ============================================================
print("Load codebook...", flush=True)

codebook_np = np.load(CODEBOOK_PATH).astype(np.float32)
assert codebook_np.shape == (num_code, dim), f"codebook shape mismatch: {codebook_np.shape}"


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
# 8. Allocate dequant buffers once
# ============================================================
print("\nAllocate vq_dequant buffers once...", flush=True)

vq_codebook_buf = None
vq_idx_buf = None
vq_zq_buf = None
dec_out_cma = None

try:
    t0 = time.perf_counter()

    vq_codebook_buf = allocate(shape=(num_code, dim), dtype=np.float32, cacheable=1)
    vq_idx_buf = allocate(shape=(num_vectors,), dtype=np.uint16, cacheable=1)
    vq_zq_buf = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)

    # Decoder output 建议复用，避免每张图 allocate/free
    dec_out_cma = allocate(shape=expected_dec_out, dtype=np.int8, cacheable=0)

    alloc_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  alloc time       = {alloc_ms:.2f} ms", flush=True)
    print(f"  codebook addr    = 0x{vq_codebook_buf.device_address:016X}", flush=True)
    print(f"  vq_idx addr      = 0x{vq_idx_buf.device_address:016X}", flush=True)
    print(f"  vq_zq addr       = 0x{vq_zq_buf.device_address:016X}", flush=True)
    print(f"  dec_out_cma addr = 0x{dec_out_cma.device_address:016X}", flush=True)

    # codebook 只同步一次
    vq_codebook_buf[:] = codebook_np
    vq_codebook_buf.sync_to_device()

    set_u64(vq_dequant.mmio, 0x1C, 0x20, vq_codebook_buf.device_address)
    set_u64(vq_dequant.mmio, 0x10, 0x14, vq_idx_buf.device_address)
    set_u64(vq_dequant.mmio, 0x28, 0x2C, vq_zq_buf.device_address)

    write_float(vq_dequant.mmio, 0x34, dec_scale_inv)

    print("vq_dequant regs:", flush=True)
    print("  dec_scale_inv =", read_float(vq_dequant.mmio, 0x34), flush=True)

    # ========================================================
    # 9. idx -> vq_dequant -> zq in memory -> Decoder -> recon
    # ========================================================
    read_idx_times = []
    deq_copy_sync_times = []
    deq_times = []
    zq_copy_times = []
    dec_times = []
    lut_times = []
    save_times = []

    print("\n[Pipeline] idx -> vq_dequant -> Decoder -> recon", flush=True)

    t_all = time.perf_counter()

    for img_id, f in enumerate(idx_files):
        if VERBOSE:
            print(f"\n[Frame] img={img_id}, idx_file={f}", flush=True)

        idx_path = os.path.join(IDX_DIR, f)

        # ----------------------------------------------------
        # Read idx
        # ----------------------------------------------------
        t0 = time.perf_counter()

        idx_np = np.fromfile(idx_path, dtype=np.uint16)

        read_idx_ms = (time.perf_counter() - t0) * 1000.0
        read_idx_times.append(read_idx_ms)

        if idx_np.shape != (num_vectors,):
            raise ValueError(f"idx shape mismatch: {idx_np.shape}, expected=({num_vectors},)")

        if CHECK_RANGE:
            idx_min = int(idx_np.min())
            idx_max = int(idx_np.max())

            if VERBOSE:
                print(f"  idx range=[{idx_min},{idx_max}]", flush=True)

            if idx_min < 0 or idx_max >= num_code:
                raise RuntimeError(f"Invalid idx range: [{idx_min},{idx_max}]")

        # ----------------------------------------------------
        # Copy idx to CMA + sync
        # ----------------------------------------------------
        t0 = time.perf_counter()

        vq_idx_buf[:] = idx_np
        vq_idx_buf.sync_to_device()

        # 如果 vq_dequant 对 vq_zq_buf 每个元素都会完整写出，
        # 不需要每帧清零 vq_zq_buf。
        # 原代码中有：
        #   vq_zq_buf[:] = 0
        #   vq_zq_buf.sync_to_device()
        # 正式运行建议删除，减少一次 8.29 MB 写和 sync。
        set_u64(vq_dequant.mmio, 0x10, 0x14, vq_idx_buf.device_address)
        set_u64(vq_dequant.mmio, 0x28, 0x2C, vq_zq_buf.device_address)

        deq_copy_sync_ms = (time.perf_counter() - t0) * 1000.0
        deq_copy_sync_times.append(deq_copy_sync_ms)

        # ----------------------------------------------------
        # Run vq_dequant
        # ----------------------------------------------------
        t0 = time.perf_counter()

        start_and_wait_old_style(vq_dequant.mmio, timeout_s=VQ_TIMEOUT_S)

        deq_ms = (time.perf_counter() - t0) * 1000.0
        deq_times.append(deq_ms)

        # ----------------------------------------------------
        # Sync zq from device and build Decoder input
        # ----------------------------------------------------
        t0 = time.perf_counter()

        vq_zq_buf.sync_from_device()

        # 注意：
        # 这里不保存 zq_npy，只在内存中构造 Decoder 输入。
        # VART Decoder 输入一般接受普通 numpy ndarray。
        zq_np = np.asarray(vq_zq_buf).reshape(expected_dec_in)

        # 如果担心 VART 对 PYNQ buffer view 不稳定，可以改成：
        # zq_np = np.array(vq_zq_buf.reshape(expected_dec_in), dtype=np.int8, copy=True)
        # 但这样会多一次 8.29 MB 拷贝。
        zq_np = np.ascontiguousarray(zq_np.astype(np.int8, copy=False))

        zq_copy_ms = (time.perf_counter() - t0) * 1000.0
        zq_copy_times.append(zq_copy_ms)

        if CHECK_RANGE and VERBOSE:
            print("  zq range:", int(zq_np.min()), int(zq_np.max()), flush=True)

        # ----------------------------------------------------
        # Run Decoder
        # ----------------------------------------------------
        t0 = time.perf_counter()

        # cacheable=0 的输出一般不需要每帧清零
        job_id = dec_runner.execute_async([zq_np], [dec_out_cma])
        dec_runner.wait(job_id)

        dec_ms = (time.perf_counter() - t0) * 1000.0
        dec_times.append(dec_ms)

        # ----------------------------------------------------
        # LUT postprocess
        # ----------------------------------------------------
        t0 = time.perf_counter()

        recon_img = post_lut[dec_out_cma[0].view(np.uint8)]

        lut_ms = (time.perf_counter() - t0) * 1000.0
        lut_times.append(lut_ms)

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------
        save_path = os.path.join(RES_DIR, f'recon_{img_id:04d}.png')

        t0 = time.perf_counter()

        cv2.imwrite(save_path, cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR))

        save_ms = (time.perf_counter() - t0) * 1000.0
        save_times.append(save_ms)

        if VERBOSE:
            print(
                f"  read_idx={read_idx_ms:.2f} ms, "
                f"deq_copy={deq_copy_sync_ms:.2f} ms, "
                f"deq={deq_ms:.2f} ms, "
                f"zq_sync={zq_copy_ms:.2f} ms, "
                f"dec={dec_ms:.2f} ms, "
                f"lut={lut_ms:.2f} ms, "
                f"save={save_ms:.2f} ms",
                flush=True
            )

        del idx_np
        del zq_np
        del recon_img

    total_s = time.perf_counter() - t_all
    total_ms = total_s * 1000.0
    fps = num_imgs / total_s if total_s > 0 else 0.0

    print("\n" + "=" * 80, flush=True)
    print("3.run_no_zq_save.py finished.", flush=True)
    print("=" * 80, flush=True)
    print(f"  frames processed        = {num_imgs}", flush=True)
    print(f"  total wall time         = {total_ms:.2f} ms", flush=True)
    print(f"  FPS                     = {fps:.2f}", flush=True)

    print_avg("read_idx", read_idx_times)
    print_avg("deq_copy_sync", deq_copy_sync_times)
    print_avg("vq_dequant", deq_times)
    print_avg("zq_sync/view", zq_copy_times)
    print_avg("Decoder", dec_times)
    print_avg("LUT", lut_times)
    print_avg("save_png", save_times)

    if len(deq_times) > 0 and len(dec_times) > 0:
        stage_ms = (
            np.mean(read_idx_times) +
            np.mean(deq_copy_sync_times) +
            np.mean(deq_times) +
            np.mean(zq_copy_times) +
            np.mean(dec_times) +
            np.mean(lut_times) +
            np.mean(save_times)
        )
        print(f"  summed stage avg        = {stage_ms:.2f} ms", flush=True)
        print(f"  summed-stage FPS        = {1000.0 / stage_ms:.2f}", flush=True)

    print("  recon saved to          :", RES_DIR, flush=True)
    print("=" * 80, flush=True)

finally:
    print("\nCleanup...", flush=True)

    try:
        del dec_runner
    except Exception:
        pass

    free_buf(vq_codebook_buf, "vq_codebook_buf")
    free_buf(vq_idx_buf, "vq_idx_buf")
    free_buf(vq_zq_buf, "vq_zq_buf")
    free_buf(dec_out_cma, "dec_out_cma")

    gc.collect()

    print("Decoder runner released.", flush=True)