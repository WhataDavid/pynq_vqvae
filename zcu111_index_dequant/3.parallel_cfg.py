#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import gc
import struct
import faulthandler
import threading
import queue
import traceback

import numpy as np
import cv2
from pynq import allocate
faulthandler.enable()

import cfg

cfg.setup_python_paths()

from pynq_dpu import DpuOverlay
import vart
import xir


# ============================================================
# 0. Local paths from cfg.py
# ============================================================
cfg.ensure_output_dirs()

WORK_DIR = cfg.WORK_DIR
PL_DIR = cfg.PL_DIR
BIT_PATH = cfg.BIT_PATH
CODEBOOK_PATH = cfg.CODEBOOK_PATH
DEC_XMODEL = cfg.DEC_XMODEL
RES_DIR = cfg.RES_DIR
IDX_DIR = cfg.IDX_DIR


# ============================================================
# 1. Shape / scale
# ============================================================
TARGET_W, TARGET_H = 768, 512
LATENT_W, LATENT_H = 192, 128

num_vectors = LATENT_W * LATENT_H      # 24576
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

# benchmark 时建议 False，避免 min/max 扫描影响 FPS。
CHECK_RANGE = False

# 如果只想处理前 10 个 idx 文件，设为 10。
# 如果为 None，则处理 IDX_DIR 下全部 idx_*.bin。
MAX_FILES = None

VQ_TIMEOUT_S = 60.0

# 队列深度
IDX_Q_DEPTH = 4
DEQ_TO_DEC_Q_DEPTH = 2
SAVE_Q_DEPTH = 2

# zq ping-pong buffer 数量。
# 每个 zq buffer 大约 8.29 MB。
# 2 通常足够：一个给 Decoder，另一个给 dequant 写。
NUM_ZQ_BUFS = 2

# 当前 block design 中的 vq_dequant IP 名称；保留在脚本内，便于按阶段单独调整。
VQ_DEQUANT_NAMES = ('vq_dequant_1', 'vq_dequant_0')


def get_overlay_ip(overlay, candidate_names):
    for name in candidate_names:
        if hasattr(overlay, name):
            return getattr(overlay, name)
    raise RuntimeError(f"Cannot find any IP instance from: {candidate_names}")


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


def print_avg(stats, name):
    arr = stats.get(name, [])
    if len(arr) == 0:
        print(f"  avg {name:<24} = N/A", flush=True)
    else:
        print(f"  avg {name:<24} = {np.mean(arr):.2f} ms", flush=True)


def safe_put_none(q):
    try:
        q.put(None, timeout=1.0)
    except Exception:
        pass


# ============================================================
# 4. Load overlay
# ============================================================
print("Load overlay...", flush=True)
overlay = DpuOverlay(BIT_PATH)

print("Overlay IPs:", flush=True)
for k in overlay.ip_dict.keys():
    print(" ", k, flush=True)

vq_dequant = get_overlay_ip(overlay, VQ_DEQUANT_NAMES)

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
# 8. Allocate shared buffers
# ============================================================
print("\nAllocate shared buffers...", flush=True)

vq_codebook_buf = None
vq_idx_buf = None
zq_bufs = []
dec_out_cma = None

try:
    t0 = time.perf_counter()

    vq_codebook_buf = allocate(shape=(num_code, dim), dtype=np.float32, cacheable=1)
    vq_idx_buf = allocate(shape=(num_vectors,), dtype=np.uint16, cacheable=1)

    zq_bufs = [
        allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)
        for _ in range(NUM_ZQ_BUFS)
    ]

    # Decoder output 只在主线程 Decoder 中使用，复用即可。
    dec_out_cma = allocate(shape=expected_dec_out, dtype=np.int8, cacheable=0)

    alloc_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  alloc time       = {alloc_ms:.2f} ms", flush=True)
    print(f"  codebook addr    = 0x{vq_codebook_buf.device_address:016X}", flush=True)
    print(f"  vq_idx addr      = 0x{vq_idx_buf.device_address:016X}", flush=True)

    for i, b in enumerate(zq_bufs):
        print(f"  zq_buf[{i}] addr  = 0x{b.device_address:016X}", flush=True)

    print(f"  dec_out_cma addr = 0x{dec_out_cma.device_address:016X}", flush=True)

    # codebook 只同步一次
    vq_codebook_buf[:] = codebook_np
    vq_codebook_buf.sync_to_device()

    set_u64(vq_dequant.mmio, 0x1C, 0x20, vq_codebook_buf.device_address)
    set_u64(vq_dequant.mmio, 0x10, 0x14, vq_idx_buf.device_address)
    write_float(vq_dequant.mmio, 0x34, dec_scale_inv)

    print("vq_dequant regs:", flush=True)
    print("  dec_scale_inv =", read_float(vq_dequant.mmio, 0x34), flush=True)


    # ========================================================
    # 9. Workers
    # ========================================================
    def read_idx_worker(idx_queue, err_queue, stats):
        try:
            for img_id, fname in enumerate(idx_files):
                idx_path = os.path.join(IDX_DIR, fname)

                t0 = time.perf_counter()

                idx_np = np.fromfile(idx_path, dtype=np.uint16)

                read_idx_ms = (time.perf_counter() - t0) * 1000.0
                stats["read_idx_ms"].append(read_idx_ms)

                if idx_np.shape != (num_vectors,):
                    raise ValueError(f"idx shape mismatch: {idx_np.shape}, expected=({num_vectors},)")

                if CHECK_RANGE:
                    idx_min = int(idx_np.min())
                    idx_max = int(idx_np.max())

                    if idx_min < 0 or idx_max >= num_code:
                        raise RuntimeError(f"Invalid idx range: [{idx_min},{idx_max}]")

                idx_queue.put((img_id, fname, idx_np))

            idx_queue.put(None)

        except Exception:
            err_queue.put(("read_idx_worker", traceback.format_exc()))
            safe_put_none(idx_queue)


    def dequant_worker(idx_queue, deq_to_dec_queue, free_zq_queue, err_queue, stats):
        try:
            while True:
                item = idx_queue.get()

                if item is None:
                    deq_to_dec_queue.put(None)
                    break

                img_id, fname, idx_np = item

                # 等待一个空闲 zq buffer
                t0 = time.perf_counter()
                zq_buf_id = free_zq_queue.get()
                wait_free_zq_ms = (time.perf_counter() - t0) * 1000.0
                stats["wait_free_zq_ms"].append(wait_free_zq_ms)

                zq_buf = zq_bufs[zq_buf_id]

                log(f"[DEQ] img={img_id}, zq_buf={zq_buf_id}")

                # ------------------------------------------------
                # Copy idx to CMA + sync
                # ------------------------------------------------
                t0 = time.perf_counter()

                vq_idx_buf[:] = idx_np
                vq_idx_buf.sync_to_device()

                # dequant 输出到当前空闲 zq buffer
                set_u64(vq_dequant.mmio, 0x10, 0x14, vq_idx_buf.device_address)
                set_u64(vq_dequant.mmio, 0x28, 0x2C, zq_buf.device_address)

                deq_copy_sync_ms = (time.perf_counter() - t0) * 1000.0
                stats["deq_copy_sync_ms"].append(deq_copy_sync_ms)

                # ------------------------------------------------
                # Run vq_dequant
                # ------------------------------------------------
                t0 = time.perf_counter()

                start_and_wait_old_style(vq_dequant.mmio, timeout_s=VQ_TIMEOUT_S)

                deq_ms = (time.perf_counter() - t0) * 1000.0
                stats["vq_dequant_ms"].append(deq_ms)

                # ------------------------------------------------
                # Sync zq from device
                # ------------------------------------------------
                t0 = time.perf_counter()

                zq_buf.sync_from_device()

                zq_sync_ms = (time.perf_counter() - t0) * 1000.0
                stats["zq_sync_ms"].append(zq_sync_ms)

                if CHECK_RANGE and VERBOSE:
                    print("  zq range:", int(zq_buf.min()), int(zq_buf.max()), flush=True)

                # 不复制、不保存 zq_npy，直接把 buffer id 交给 Decoder。
                deq_to_dec_queue.put((img_id, fname, zq_buf_id))

                del idx_np

        except Exception:
            err_queue.put(("dequant_worker", traceback.format_exc()))
            safe_put_none(deq_to_dec_queue)


    def save_worker(save_queue, err_queue, stats, saved_counter):
        try:
            while True:
                item = save_queue.get()

                if item is None:
                    break

                img_id, recon_img = item
                save_path = os.path.join(RES_DIR, f'recon_{img_id:04d}.png')

                t0 = time.perf_counter()

                cv2.imwrite(save_path, cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR))

                save_ms = (time.perf_counter() - t0) * 1000.0
                stats["save_png_ms"].append(save_ms)

                saved_counter["count"] += 1

                del recon_img

        except Exception:
            err_queue.put(("save_worker", traceback.format_exc()))


    # ========================================================
    # 10. Main pipeline: Decoder runs in main thread
    # ========================================================
    def run_pipeline():
        idx_queue = queue.Queue(maxsize=IDX_Q_DEPTH)
        deq_to_dec_queue = queue.Queue(maxsize=DEQ_TO_DEC_Q_DEPTH)
        save_queue = queue.Queue(maxsize=SAVE_Q_DEPTH)
        free_zq_queue = queue.Queue(maxsize=NUM_ZQ_BUFS)
        err_queue = queue.Queue()

        for i in range(NUM_ZQ_BUFS):
            free_zq_queue.put(i)

        saved_counter = {"count": 0}

        stats = {
            "read_idx_ms": [],
            "wait_free_zq_ms": [],
            "deq_copy_sync_ms": [],
            "vq_dequant_ms": [],
            "zq_sync_ms": [],
            "dec_queue_get_wait_ms": [],
            "Decoder_ms": [],
            "LUT_ms": [],
            "save_queue_put_wait_ms": [],
            "save_png_ms": [],
        }

        print(f"\n[Pipeline] Read idx -> vq_dequant -> Main Decoder -> Save: {num_imgs} frames", flush=True)

        t_all = time.perf_counter()

        t_read = threading.Thread(
            target=read_idx_worker,
            args=(idx_queue, err_queue, stats),
            name="read_idx_worker"
        )

        t_deq = threading.Thread(
            target=dequant_worker,
            args=(idx_queue, deq_to_dec_queue, free_zq_queue, err_queue, stats),
            name="dequant_worker"
        )

        t_save = threading.Thread(
            target=save_worker,
            args=(save_queue, err_queue, stats, saved_counter),
            name="save_worker"
        )

        t_read.start()
        t_deq.start()
        t_save.start()

        decoded_count = 0

        try:
            while True:
                if not err_queue.empty():
                    stage_name, err_text = err_queue.get()
                    raise RuntimeError(f"\nPipeline failed in {stage_name}:\n{err_text}")

                # ------------------------------------------------
                # Get dequantized zq buffer
                # ------------------------------------------------
                t0 = time.perf_counter()
                item = deq_to_dec_queue.get()
                dec_queue_get_wait_ms = (time.perf_counter() - t0) * 1000.0

                if item is None:
                    break

                img_id, fname, zq_buf_id = item
                stats["dec_queue_get_wait_ms"].append(dec_queue_get_wait_ms)

                zq_buf = zq_bufs[zq_buf_id]

                # ------------------------------------------------
                # Build Decoder input view
                # ------------------------------------------------
                zq_np = np.asarray(zq_buf).reshape(expected_dec_in)
                zq_np = np.ascontiguousarray(zq_np.astype(np.int8, copy=False))

                # ------------------------------------------------
                # Run Decoder in main thread
                # ------------------------------------------------
                t0 = time.perf_counter()

                job_id = dec_runner.execute_async([zq_np], [dec_out_cma])
                dec_runner.wait(job_id)

                dec_ms = (time.perf_counter() - t0) * 1000.0
                stats["Decoder_ms"].append(dec_ms)

                # Decoder 已经读取完 zq_buf，可以还给 dequant_worker 复用。
                free_zq_queue.put(zq_buf_id)

                # ------------------------------------------------
                # LUT postprocess
                # ------------------------------------------------
                t0 = time.perf_counter()

                recon_img = post_lut[dec_out_cma[0].view(np.uint8)]

                lut_ms = (time.perf_counter() - t0) * 1000.0
                stats["LUT_ms"].append(lut_ms)

                # ------------------------------------------------
                # Send image to save worker
                # ------------------------------------------------
                t0 = time.perf_counter()
                save_queue.put((img_id, recon_img))
                save_queue_put_wait_ms = (time.perf_counter() - t0) * 1000.0
                stats["save_queue_put_wait_ms"].append(save_queue_put_wait_ms)

                decoded_count += 1

                if VERBOSE:
                    print(
                        f"[DEC] img={img_id}, "
                        f"dec={dec_ms:.2f} ms, "
                        f"lut={lut_ms:.2f} ms, "
                        f"zq_buf={zq_buf_id}",
                        flush=True
                    )

                del zq_np

        finally:
            safe_put_none(save_queue)

            t_read.join()
            t_deq.join()
            t_save.join()

        total_s = time.perf_counter() - t_all
        total_ms = total_s * 1000.0

        if not err_queue.empty():
            stage_name, err_text = err_queue.get()
            raise RuntimeError(f"\nPipeline failed in {stage_name}:\n{err_text}")

        saved_count = saved_counter["count"]
        fps = saved_count / total_s if total_s > 0 else 0.0

        print("\n" + "=" * 80, flush=True)
        print("3.parallel_no_zq_save.py finished.", flush=True)
        print("=" * 80, flush=True)
        print(f"  frames decoded             = {decoded_count}/{num_imgs}", flush=True)
        print(f"  recon saved                = {saved_count}/{num_imgs}", flush=True)
        print(f"  total wall time            = {total_ms:.2f} ms", flush=True)
        print(f"  pipeline FPS               = {fps:.2f}", flush=True)

        print_avg(stats, "read_idx_ms")
        print_avg(stats, "wait_free_zq_ms")
        print_avg(stats, "deq_copy_sync_ms")
        print_avg(stats, "vq_dequant_ms")
        print_avg(stats, "zq_sync_ms")
        print_avg(stats, "dec_queue_get_wait_ms")
        print_avg(stats, "Decoder_ms")
        print_avg(stats, "LUT_ms")
        print_avg(stats, "save_queue_put_wait_ms")
        print_avg(stats, "save_png_ms")

        if len(stats["vq_dequant_ms"]) > 0:
            deq_stage_ms = (
                np.mean(stats["deq_copy_sync_ms"]) +
                np.mean(stats["vq_dequant_ms"]) +
                np.mean(stats["zq_sync_ms"])
            )
            print(f"  dequant stage avg          = {deq_stage_ms:.2f} ms", flush=True)
            print(f"  dequant stage FPS          = {1000.0 / deq_stage_ms:.2f}", flush=True)

        if len(stats["Decoder_ms"]) > 0:
            dec_stage_ms = np.mean(stats["Decoder_ms"]) + np.mean(stats["LUT_ms"])
            print(f"  Decoder+LUT stage avg      = {dec_stage_ms:.2f} ms", flush=True)
            print(f"  Decoder+LUT stage FPS      = {1000.0 / dec_stage_ms:.2f}", flush=True)

        if len(stats["vq_dequant_ms"]) > 0 and len(stats["Decoder_ms"]) > 0:
            deq_stage_ms = (
                np.mean(stats["deq_copy_sync_ms"]) +
                np.mean(stats["vq_dequant_ms"]) +
                np.mean(stats["zq_sync_ms"])
            )
            dec_stage_ms = np.mean(stats["Decoder_ms"]) + np.mean(stats["LUT_ms"])
            save_stage_ms = np.mean(stats["save_png_ms"]) if len(stats["save_png_ms"]) > 0 else 0.0

            max_stage_ms = max(deq_stage_ms, dec_stage_ms, save_stage_ms)
            print(f"  save stage avg             = {save_stage_ms:.2f} ms", flush=True)
            print(f"  max-stage FPS approx       = {1000.0 / max_stage_ms:.2f}", flush=True)

        print("  recon saved to             :", RES_DIR, flush=True)
        print("=" * 80, flush=True)

        return stats


    # ========================================================
    # 11. Run
    # ========================================================
    stats = run_pipeline()

finally:
    print("\nCleanup...", flush=True)

    try:
        del dec_runner
    except Exception:
        pass

    free_buf(vq_codebook_buf, "vq_codebook_buf")
    free_buf(vq_idx_buf, "vq_idx_buf")

    for i, b in enumerate(zq_bufs):
        free_buf(b, f"zq_buf[{i}]")

    free_buf(dec_out_cma, "dec_out_cma")

    gc.collect()

    print("Decoder runner released.", flush=True)