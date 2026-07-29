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
PRE_DIR = cfg.PRE_DIR
CODEBOOK_PATH = getattr(cfg, 'VQ_CODEBOOK_PATH', cfg.CODEBOOK_PATH)
ENC_XMODEL = cfg.ENC_XMODEL
RES_DIR = cfg.RES_DIR
IDX_DIR = cfg.IDX_DIR


# ============================================================
# 1. Shape / scale
# ============================================================
TARGET_W, TARGET_H = 960, 540
LATENT_W, LATENT_H = 240, 135

num_vectors = LATENT_W * LATENT_H      # 32400
dim = 64
num_code = 512

expected_enc_in = (1, TARGET_H, TARGET_W, 3)
expected_enc_out = (1, LATENT_H, LATENT_W, dim)

enc_in_scale = 0.015625
enc_out_scale = 0.015625

dec_in_scale = 0.03125
dec_scale_inv = 1.0 / dec_in_scale


# ============================================================
# 2. Runtime options
# ============================================================
# Benchmark 时建议 False，避免 stdout flush 和 min/max 扫描拖慢 FPS。
VERBOSE = False
CHECK_RANGE = False

# 如果只有 10 张图，首尾流水线开销占比很大。
# 做稳态 benchmark 时可改成 5 或 10，即把 data_files 重复多轮。
# 实际保存文件名会自动使用 img_id，不会覆盖。
REPEAT_DATASET = 1

# 队列深度：
# READ_Q_DEPTH 控制预读输入数量。
# ENC_TO_VQ_Q_DEPTH 控制 Encoder 输出积压数量。
# 960x540 的 enc_out_np 约 2.07 MB/张，不建议太大。
READ_Q_DEPTH = 2
ENC_TO_VQ_Q_DEPTH = 2

VQ_TIMEOUT_S = 60.0

# 当前 block design 中的 VQ IP 名称；保留在脚本内，便于按阶段单独调整。
VQ_ACCEL_NAMES = ('vq_accel_1', 'vq_accel_0')


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
        print(f"  avg {name:<22} = N/A", flush=True)
    else:
        print(f"  avg {name:<22} = {np.mean(arr):.2f} ms", flush=True)


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

vq_accel = get_overlay_ip(overlay, VQ_ACCEL_NAMES)

print("vq_accel register_map:", flush=True)
print(vq_accel.register_map, flush=True)


# ============================================================
# 5. Create Encoder runner
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
# 6. Data files / codebook
# ============================================================
if not os.path.exists(PRE_DIR):
    raise RuntimeError(f"PRE_DIR does not exist: {PRE_DIR}")

base_data_files = sorted([f for f in os.listdir(PRE_DIR) if f.endswith('.npy')])

if len(base_data_files) == 0:
    raise RuntimeError(f"No npy files found in {PRE_DIR}")

data_files = base_data_files * REPEAT_DATASET
num_imgs = len(data_files)

print(f"Found {len(base_data_files)} images in {PRE_DIR}", flush=True)
print(f"Total benchmark frames: {num_imgs}", flush=True)

codebook_np = np.load(CODEBOOK_PATH).astype(np.float32)
assert codebook_np.shape == (num_code, dim), f"codebook shape mismatch: {codebook_np.shape}"


# ============================================================
# 7. Input preparation
# ============================================================
def load_and_prepare_input(fname):
    path = os.path.join(PRE_DIR, fname)

    t0 = time.perf_counter()
    data = np.load(path)
    read_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()

    if data.shape == (TARGET_H, TARGET_W, 3):
        data_hwc = data
    elif data.shape == (1, TARGET_H, TARGET_W, 3):
        data_hwc = data[0]
    else:
        raise ValueError(f"Input shape mismatch: {data.shape}")

    if data_hwc.dtype == np.int8:
        enc_input_np = np.ascontiguousarray(data_hwc[np.newaxis])
    else:
        enc_input_np = np.clip(
            np.round(data_hwc.astype(np.float32) / enc_in_scale),
            -128,
            127
        ).astype(np.int8)[np.newaxis]
        enc_input_np = np.ascontiguousarray(enc_input_np)

    if enc_input_np.shape != expected_enc_in:
        raise ValueError(f"enc_input_np shape mismatch: {enc_input_np.shape}")

    prep_ms = (time.perf_counter() - t0) * 1000.0

    del data

    return enc_input_np, read_ms, prep_ms


# ============================================================
# 8. Read worker
# ============================================================
def read_worker(read_queue, err_queue, stats):
    try:
        for img_id, fname in enumerate(data_files):
            enc_input_np, read_ms, prep_ms = load_and_prepare_input(fname)

            stats["read_ms"].append(read_ms)
            stats["prep_ms"].append(prep_ms)

            if CHECK_RANGE:
                in_min = int(enc_input_np.min())
                in_max = int(enc_input_np.max())
                log(f"[READ] img={img_id}, range=[{in_min},{in_max}]")

            read_queue.put((img_id, fname, enc_input_np))

        read_queue.put(None)

    except Exception:
        err_queue.put(("read_worker", traceback.format_exc()))
        safe_put_none(read_queue)


# ============================================================
# 9. Background VQ + Save worker
# ============================================================
def vq_save_worker(enc_to_vq_queue, err_queue, stats, saved_counter):
    vq_codebook_buf = None
    vq_in_buf = None
    vq_idx_buf = None

    try:
        print("\n[VQ] Allocate vq_accel buffers once...", flush=True)
        t0 = time.perf_counter()

        vq_codebook_buf = allocate(shape=(num_code, dim), dtype=np.float32, cacheable=1)
        vq_in_buf = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)
        vq_idx_buf = allocate(shape=(num_vectors,), dtype=np.uint16, cacheable=1)

        vq_init_ms = (time.perf_counter() - t0) * 1000.0
        stats["vq_init_ms"].append(vq_init_ms)

        print(f"[VQ] buffer init time: {vq_init_ms:.2f} ms", flush=True)
        print(f"  codebook addr=0x{vq_codebook_buf.device_address:016X}", flush=True)
        print(f"  vq_in    addr=0x{vq_in_buf.device_address:016X}", flush=True)
        print(f"  vq_idx   addr=0x{vq_idx_buf.device_address:016X}", flush=True)

        # codebook 只同步一次
        vq_codebook_buf[:] = codebook_np
        vq_codebook_buf.sync_to_device()

        set_u64(vq_accel.mmio, 0x1C, 0x20, vq_codebook_buf.device_address)
        write_float(vq_accel.mmio, 0x34, enc_out_scale)
        write_float(vq_accel.mmio, 0x3C, dec_scale_inv)

        print("[VQ] static regs:", flush=True)
        print("  enc_scale     =", read_float(vq_accel.mmio, 0x34), flush=True)
        print("  dec_scale_inv =", read_float(vq_accel.mmio, 0x3C), flush=True)

        while True:
            item = enc_to_vq_queue.get()

            if item is None:
                break

            img_id, fname, enc_out_np = item

            log(f"[VQ] img={img_id}, file={fname}")

            # ------------------------------------------------
            # Copy Encoder output to CMA + sync to device
            # ------------------------------------------------
            t0 = time.perf_counter()

            vq_in_buf[:] = enc_out_np.reshape(num_vectors, dim)
            vq_in_buf.sync_to_device()

            # 优化：如果 vq_accel 对 out_idx 每个元素都会完整写出，
            # 则不需要每帧清零 out_idx，也不需要 sync_to_device。
            # 原代码：
            #   vq_idx_buf[:] = 0
            #   vq_idx_buf.sync_to_device()

            set_u64(vq_accel.mmio, 0x10, 0x14, vq_in_buf.device_address)
            set_u64(vq_accel.mmio, 0x28, 0x2C, vq_idx_buf.device_address)

            vq_copy_sync_ms = (time.perf_counter() - t0) * 1000.0
            stats["vq_copy_sync_ms"].append(vq_copy_sync_ms)

            # ------------------------------------------------
            # Run VQ accelerator
            # ------------------------------------------------
            t0 = time.perf_counter()

            start_and_wait_old_style(vq_accel.mmio, timeout_s=VQ_TIMEOUT_S)

            vq_hw_ms = (time.perf_counter() - t0) * 1000.0
            stats["vq_hw_ms"].append(vq_hw_ms)

            # ------------------------------------------------
            # Sync output idx from device
            # ------------------------------------------------
            t0 = time.perf_counter()

            vq_idx_buf.sync_from_device()

            idx_copy_ms = (time.perf_counter() - t0) * 1000.0
            stats["idx_copy_ms"].append(idx_copy_ms)

            # 可选检查：benchmark 时建议关闭。
            if CHECK_RANGE:
                idx_min = int(vq_idx_buf.min())
                idx_max = int(vq_idx_buf.max())
                log(f"[VQ] idx range=[{idx_min},{idx_max}]")

                if idx_min < 0 or idx_max >= num_code:
                    raise RuntimeError(f"Invalid VQ index range: [{idx_min},{idx_max}]")

            # ------------------------------------------------
            # Save idx
            # ------------------------------------------------
            idx_path = os.path.join(IDX_DIR, f'idx_{img_id:04d}.bin')

            t0 = time.perf_counter()

            # 直接保存 PYNQ buffer 内容，避免 np.array(..., copy=True)。
            # 若你的环境中 memoryview 不兼容，可改回：
            #   fp.write(np.asarray(vq_idx_buf, dtype=np.uint16).tobytes())
            with open(idx_path, 'wb') as fp:
                fp.write(memoryview(vq_idx_buf))

            idx_save_ms = (time.perf_counter() - t0) * 1000.0
            stats["idx_save_ms"].append(idx_save_ms)

            saved_counter["count"] += 1

            if VERBOSE:
                print(
                    f"[VQ/SAVE] img={img_id}, "
                    f"copy_sync={vq_copy_sync_ms:.2f} ms, "
                    f"vq={vq_hw_ms:.2f} ms, "
                    f"sync_idx={idx_copy_ms:.2f} ms, "
                    f"save={idx_save_ms:.2f} ms",
                    flush=True
                )

            del enc_out_np

    except Exception:
        err_queue.put(("vq_save_worker", traceback.format_exc()))

    finally:
        print("\n[VQ] Free vq_accel buffers...", flush=True)
        free_buf(vq_codebook_buf, "vq_codebook_buf")
        free_buf(vq_in_buf, "vq_in_buf")
        free_buf(vq_idx_buf, "vq_idx_buf")


# ============================================================
# 10. Main pipeline
# ============================================================
def run_pipeline():
    read_queue = queue.Queue(maxsize=READ_Q_DEPTH)
    enc_to_vq_queue = queue.Queue(maxsize=ENC_TO_VQ_Q_DEPTH)
    err_queue = queue.Queue()
    saved_counter = {"count": 0}

    stats = {
        "read_ms": [],
        "prep_ms": [],
        "read_queue_get_wait_ms": [],
        "enc_hw_ms": [],
        "enc_to_vq_put_wait_ms": [],
        "vq_init_ms": [],
        "vq_copy_sync_ms": [],
        "vq_hw_ms": [],
        "idx_copy_ms": [],
        "idx_save_ms": [],
    }

    print(f"\n[Pipeline] Read worker -> Main Encoder -> VQ/Save worker: {num_imgs} frames", flush=True)

    t_all = time.perf_counter()

    t_read = threading.Thread(
        target=read_worker,
        args=(read_queue, err_queue, stats),
        name="read_worker"
    )

    t_vq = threading.Thread(
        target=vq_save_worker,
        args=(enc_to_vq_queue, err_queue, stats, saved_counter),
        name="vq_save_worker"
    )

    t_read.start()
    t_vq.start()

    encoded_count = 0

    try:
        while True:
            if not err_queue.empty():
                stage_name, err_text = err_queue.get()
                raise RuntimeError(f"\nPipeline failed in {stage_name}:\n{err_text}")

            # ------------------------------------------------
            # Get prepared input
            # ------------------------------------------------
            t0 = time.perf_counter()
            item = read_queue.get()
            read_queue_get_wait_ms = (time.perf_counter() - t0) * 1000.0

            if item is None:
                break

            img_id, fname, enc_input_np = item
            stats["read_queue_get_wait_ms"].append(read_queue_get_wait_ms)

            log(f"[ENC-MAIN] img={img_id}, file={fname}")

            # ------------------------------------------------
            # Run Encoder in main thread
            # ------------------------------------------------
            enc_out_np = np.empty(expected_enc_out, dtype=np.int8, order='C')

            t0 = time.perf_counter()

            job_id = enc_runner.execute_async([enc_input_np], [enc_out_np])
            enc_runner.wait(job_id)

            enc_hw_ms = (time.perf_counter() - t0) * 1000.0
            stats["enc_hw_ms"].append(enc_hw_ms)

            if CHECK_RANGE:
                out_min = int(enc_out_np.min())
                out_max = int(enc_out_np.max())
                log(f"[ENC] output range=[{out_min},{out_max}]")

            # ------------------------------------------------
            # Send to VQ worker
            # ------------------------------------------------
            t0 = time.perf_counter()
            enc_to_vq_queue.put((img_id, fname, enc_out_np))
            enc_to_vq_put_wait_ms = (time.perf_counter() - t0) * 1000.0
            stats["enc_to_vq_put_wait_ms"].append(enc_to_vq_put_wait_ms)

            encoded_count += 1

            if VERBOSE:
                print(
                    f"[ENC] img={img_id}, "
                    f"enc={enc_hw_ms:.2f} ms, "
                    f"read_wait={read_queue_get_wait_ms:.2f} ms, "
                    f"vq_put_wait={enc_to_vq_put_wait_ms:.2f} ms",
                    flush=True
                )

            del enc_input_np

    finally:
        safe_put_none(enc_to_vq_queue)
        t_read.join()
        t_vq.join()

    total_s = time.perf_counter() - t_all
    total_ms = total_s * 1000.0

    if not err_queue.empty():
        stage_name, err_text = err_queue.get()
        raise RuntimeError(f"\nPipeline failed in {stage_name}:\n{err_text}")

    saved_count = saved_counter["count"]
    fps = saved_count / total_s if total_s > 0 else 0.0

    print("\n" + "=" * 80, flush=True)
    print("Optimized pipeline finished.", flush=True)
    print("=" * 80, flush=True)
    print(f"  frames encoded             = {encoded_count}/{num_imgs}", flush=True)
    print(f"  idx saved                  = {saved_count}/{num_imgs}", flush=True)
    print(f"  total wall time            = {total_ms:.2f} ms", flush=True)
    print(f"  pipeline FPS               = {fps:.2f}", flush=True)

    print_avg(stats, "read_ms")
    print_avg(stats, "prep_ms")
    print_avg(stats, "read_queue_get_wait_ms")
    print_avg(stats, "enc_hw_ms")
    print_avg(stats, "enc_to_vq_put_wait_ms")
    print_avg(stats, "vq_copy_sync_ms")
    print_avg(stats, "vq_hw_ms")
    print_avg(stats, "idx_copy_ms")
    print_avg(stats, "idx_save_ms")

    if len(stats["vq_init_ms"]) > 0:
        print(f"  vq buffer init             = {stats['vq_init_ms'][0]:.2f} ms", flush=True)

    if len(stats["enc_hw_ms"]) > 0:
        enc_fps = 1000.0 / np.mean(stats["enc_hw_ms"])
        print(f"  Encoder hw-only FPS        = {enc_fps:.2f}", flush=True)

    if len(stats["vq_hw_ms"]) > 0:
        vq_fps = 1000.0 / np.mean(stats["vq_hw_ms"])
        print(f"  vq_accel hw-only FPS       = {vq_fps:.2f}", flush=True)

    if len(stats["enc_hw_ms"]) > 0 and len(stats["vq_hw_ms"]) > 0:
        vq_stage_ms = (
            np.mean(stats["vq_copy_sync_ms"]) +
            np.mean(stats["vq_hw_ms"]) +
            np.mean(stats["idx_copy_ms"]) +
            np.mean(stats["idx_save_ms"])
        )

        enc_stage_ms = np.mean(stats["enc_hw_ms"])

        print(f"  Encoder stage avg          = {enc_stage_ms:.2f} ms", flush=True)
        print(f"  VQ+Save stage avg          = {vq_stage_ms:.2f} ms", flush=True)
        print(f"  max-stage FPS approx       = {1000.0 / max(enc_stage_ms, vq_stage_ms):.2f}", flush=True)

    print("  IDX saved to               :", IDX_DIR, flush=True)
    print("=" * 80, flush=True)

    return stats


# ============================================================
# 11. Main
# ============================================================
if __name__ == "__main__":
    try:
        stats = run_pipeline()

    finally:
        try:
            del enc_runner
        except Exception:
            pass

        gc.collect()
        print("Encoder runner released.", flush=True)