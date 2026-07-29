#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['XILINX_XRT'] = '/usr'
FIRMWARE_DIR = '/lib/firmware'
os.makedirs(FIRMWARE_DIR, exist_ok=True)

import sys
import time
import struct
import json

import numpy as np
import cv2
from pynq import allocate

import cfg
cfg.setup_python_paths()
cfg.ensure_output_dirs()

from pynq_dpu import DpuOverlay
import vart
import xir


# ============================================================
# Constants / paths from cfg.py
# ============================================================
BIT_PATH = cfg.BIT_PATH
CODEBOOK_PATH = cfg.CODEBOOK_PATH
VQ_CODEBOOK_PATH = getattr(cfg, 'VQ_CODEBOOK_PATH', cfg.CODEBOOK_PATH)
ENC_XMODEL = cfg.ENC_XMODEL
DEC_XMODEL = cfg.DEC_XMODEL

# Keep test.py original single-image encode/decode command logic.
# Only replace tensor sizes and file/env paths for 1280x720.
enc_out_scale = 0.015625
enc_in_scale = 0.015625

dec_in_scale = 0.03125
dec_out_scale = 0.0078125
dec_scale_inv = 1.0 / dec_in_scale

IMG_H = 720
IMG_W = 1280
LATENT_H = 180
LATENT_W = 320

num_vectors = LATENT_H * LATENT_W
dim = 64
num_code = 512

expected_enc_in = (1, IMG_H, IMG_W, 3)
expected_enc_out = (1, LATENT_H, LATENT_W, dim)
expected_dec_in = (1, LATENT_H, LATENT_W, dim)
expected_dec_out = (1, IMG_H, IMG_W, 3)

VQ_TIMEOUT_S = 60.0

VQ_ACCEL_NAMES = ('vq_accel_1', 'vq_accel_0')
VQ_DEQUANT_NAMES = ('vq_dequant_1', 'vq_dequant_0')


def get_overlay_ip(overlay, candidate_names):
    for name in candidate_names:
        if hasattr(overlay, name):
            return getattr(overlay, name)
    raise RuntimeError(f"Cannot find any IP instance from: {candidate_names}")


# ============================================================
# Preprocessing
# ============================================================
def preprocess_png(png_path):
    t0 = time.perf_counter()

    img_bgr = cv2.imread(png_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {png_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    input_fp32 = img_resized.astype(np.float32)
    input_fp32 = (input_fp32 / 255.0 - 0.5) / 0.5

    input_int8 = np.clip(
        np.round(input_fp32 / enc_in_scale),
        -128,
        127
    ).astype(np.int8)

    enc_input_np = np.ascontiguousarray(input_int8[np.newaxis])

    if enc_input_np.shape != expected_enc_in:
        raise ValueError(f"Encoder input shape mismatch after preprocess: {enc_input_np.shape}")

    print(f"Preprocess done: {(time.perf_counter() - t0) * 1000:.2f} ms", flush=True)
    print(f"  resized RGB shape: {img_resized.shape}", flush=True)
    print(f"  int8 range: [{int(input_int8.min())}, {int(input_int8.max())}]", flush=True)
    return enc_input_np


# ============================================================
# Helpers
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
        raise RuntimeError(f"No DPU subgraph found in {path}")

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


def build_lut(scale):
    post_lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        val_int8 = np.int8(i)
        val_fp32 = float(val_int8) * scale
        val_norm = max(0.0, min(1.0, val_fp32 * 0.5 + 0.5))
        post_lut[i] = int(val_norm * 255.0)
    return post_lut


def print_runtime_config():
    print("=" * 80, flush=True)
    print("test_cfg_1280x720.py", flush=True)
    print("=" * 80, flush=True)
    print("BIT_PATH     :", BIT_PATH, flush=True)
    print("CODEBOOK_PATH:", CODEBOOK_PATH, flush=True)
    print("ENC_XMODEL   :", ENC_XMODEL, flush=True)
    print("DEC_XMODEL   :", DEC_XMODEL, flush=True)
    print("IMG_H, IMG_W :", IMG_H, IMG_W, flush=True)
    print("LATENT_H/W   :", LATENT_H, LATENT_W, flush=True)
    print("num_vectors  :", num_vectors, flush=True)
    print("=" * 80, flush=True)


# ============================================================
# Overlay
# ============================================================
def init_overlay():
    print_runtime_config()
    print("Loading bitstream...", flush=True)
    overlay = DpuOverlay(BIT_PATH, download=True)
    print("Bitstream loaded.", flush=True)
    print("Overlay IPs:", flush=True)
    for k in overlay.ip_dict.keys():
        print(" ", k, flush=True)
    return overlay


# ============================================================
# Encode: .png -> .bin (index)
# ============================================================
def cmd_encode(input_file, output_file):
    overlay = init_overlay()

    enc_runner = None
    vq1_stage_in_buf = None
    vq1_stage_idx_buf = None
    vq_codebook_buf = None

    try:
        print("Create encoder runner...", flush=True)
        _, enc_subgraph = get_dpu_subgraph(ENC_XMODEL)
        enc_runner = vart.Runner.create_runner(enc_subgraph, "run")

        enc_in_tensors = enc_runner.get_input_tensors()
        enc_out_tensors = enc_runner.get_output_tensors()

        print("Encoder input :", tuple(enc_in_tensors[0].dims), flush=True)
        print("Encoder output:", tuple(enc_out_tensors[0].dims), flush=True)

        if tuple(enc_in_tensors[0].dims) != expected_enc_in:
            raise ValueError(f"Encoder input shape mismatch: {tuple(enc_in_tensors[0].dims)} != {expected_enc_in}")
        if tuple(enc_out_tensors[0].dims) != expected_enc_out:
            raise ValueError(f"Encoder output shape mismatch: {tuple(enc_out_tensors[0].dims)} != {expected_enc_out}")

        print("Encoder fix_point:", flush=True)
        enc_in_scales = get_fixpoint_scale(enc_in_tensors)
        enc_out_scales = get_fixpoint_scale(enc_out_tensors)
        current_enc_in_scale = enc_in_scales[0]
        current_enc_out_scale = enc_out_scales[0]

        print("Using enc_in_scale :", current_enc_in_scale, flush=True)
        print("Using enc_out_scale:", current_enc_out_scale, flush=True)

        vq_accel = get_overlay_ip(overlay, VQ_ACCEL_NAMES)

        print("Allocate buffers...", flush=True)
        vq1_stage_in_buf = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)
        vq1_stage_idx_buf = allocate(shape=(num_vectors,), dtype=np.uint16, cacheable=1)

        codebook = np.load(VQ_CODEBOOK_PATH).astype(np.float32)
        assert codebook.shape == (num_code, dim), f"VQ codebook shape mismatch: {codebook.shape}"

        vq_codebook_buf = allocate(shape=(num_code, dim), dtype=np.float32, cacheable=1)
        vq_codebook_buf[:] = codebook
        vq_codebook_buf.sync_to_device()

        set_u64(vq_accel.mmio, 0x1C, 0x20, vq_codebook_buf.device_address)
        write_float(vq_accel.mmio, 0x34, current_enc_out_scale)
        write_float(vq_accel.mmio, 0x3C, dec_scale_inv)

        print("vq_accel regs:", flush=True)
        print("  enc_scale     =", read_float(vq_accel.mmio, 0x34), flush=True)
        print("  dec_scale_inv =", read_float(vq_accel.mmio, 0x3C), flush=True)

        print(f"Load {input_file}...", flush=True)
        enc_input_np = preprocess_png(input_file)

        # If the real model input scale differs from the default, rebuild input.
        # Most current xmodels use fix_point=6 => 0.015625, so this branch normally does nothing.
        if abs(current_enc_in_scale - enc_in_scale) > 1e-12:
            img_bgr = cv2.imread(input_file)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {input_file}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
            input_fp32 = (img_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
            enc_input_np = np.ascontiguousarray(
                np.clip(np.round(input_fp32 / current_enc_in_scale), -128, 127).astype(np.int8)[np.newaxis]
            )

        enc_out_np = np.empty(expected_enc_out, dtype=np.int8, order='C')
        print("Run Encoder...", flush=True)
        t0 = time.perf_counter()
        job_id = enc_runner.execute_async([enc_input_np], [enc_out_np])
        enc_runner.wait(job_id)
        print(f"Encoder done: {(time.perf_counter() - t0) * 1000:.2f} ms", flush=True)

        enc_out_np.reshape(-1).tofile("enc_out_pynq_int8_nhwc.bin")
        print(
            "Encoder output saved -> enc_out_pynq_int8_nhwc.bin "
            f"({enc_out_np.nbytes} bytes)",
            flush=True
        )
        print(
            f"Encoder output range: "
            f"min={int(enc_out_np.min())}, "
            f"max={int(enc_out_np.max())}, "
            f"mean={float(enc_out_np.mean()):.2f}, "
            f"std={float(enc_out_np.std()):.2f}",
            flush=True
        )
        
        vq1_stage_in_buf[:] = enc_out_np.reshape(num_vectors, dim)
        vq1_stage_in_buf.sync_to_device()
        vq1_stage_idx_buf[:] = 0
        vq1_stage_idx_buf.sync_to_device()

        set_u64(vq_accel.mmio, 0x10, 0x14, vq1_stage_in_buf.device_address)
        set_u64(vq_accel.mmio, 0x28, 0x2C, vq1_stage_idx_buf.device_address)

        print("Run vq_accel...", flush=True)
        t0 = time.perf_counter()
        start_and_wait_old_style(vq_accel.mmio, timeout_s=VQ_TIMEOUT_S)
        print(f"vq_accel done: {(time.perf_counter() - t0) * 1000:.2f} ms", flush=True)

        vq1_stage_idx_buf.sync_from_device()
        idx_snapshot = np.array(vq1_stage_idx_buf, dtype=np.uint16, copy=True)

        idx_min, idx_max = int(idx_snapshot.min()), int(idx_snapshot.max())
        print(f"idx range: [{idx_min}, {idx_max}]", flush=True)
        if idx_min < 0 or idx_max >= num_code:
            raise RuntimeError(f"Invalid index range: [{idx_min}, {idx_max}]")

        output_dir = os.path.dirname(os.path.abspath(output_file))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'wb') as fp:
            fp.write(idx_snapshot.tobytes())
        print(f"Index saved -> {output_file} ({idx_snapshot.nbytes} bytes)", flush=True)

    finally:
        free_buf(vq1_stage_in_buf, "vq1_stage_in_buf")
        free_buf(vq1_stage_idx_buf, "vq1_stage_idx_buf")
        free_buf(vq_codebook_buf, "vq_codebook_buf")
        try:
            del enc_runner
        except Exception:
            pass


# ============================================================
# Decode: .bin (index) -> .png
# ============================================================
def cmd_decode(input_file, output_file):
    overlay = init_overlay()

    dec_runner = None
    vq2_stage_idx_buf = None
    vq2_stage_zq_buf = None
    vq_codebook_buf = None

    try:
        print("Create decoder runner...", flush=True)
        _, dec_subgraph = get_dpu_subgraph(DEC_XMODEL)
        dec_runner = vart.Runner.create_runner(dec_subgraph, "run")

        dec_in_tensors = dec_runner.get_input_tensors()
        dec_out_tensors = dec_runner.get_output_tensors()

        print("Decoder input :", tuple(dec_in_tensors[0].dims), flush=True)
        print("Decoder output:", tuple(dec_out_tensors[0].dims), flush=True)

        if tuple(dec_in_tensors[0].dims) != expected_dec_in:
            raise ValueError(f"Decoder input shape mismatch: {tuple(dec_in_tensors[0].dims)} != {expected_dec_in}")
        if tuple(dec_out_tensors[0].dims) != expected_dec_out:
            raise ValueError(f"Decoder output shape mismatch: {tuple(dec_out_tensors[0].dims)} != {expected_dec_out}")

        print("Decoder fix_point:", flush=True)
        dec_in_scales = get_fixpoint_scale(dec_in_tensors)
        dec_out_scales = get_fixpoint_scale(dec_out_tensors)
        current_dec_in_scale = dec_in_scales[0]
        current_dec_out_scale = dec_out_scales[0]
        current_dec_scale_inv = 1.0 / current_dec_in_scale

        print("Using dec_in_scale :", current_dec_in_scale, flush=True)
        print("Using dec_out_scale:", current_dec_out_scale, flush=True)

        vq_dequant = get_overlay_ip(overlay, VQ_DEQUANT_NAMES)

        print("Allocate buffers...", flush=True)
        vq2_stage_idx_buf = allocate(shape=(num_vectors,), dtype=np.uint16, cacheable=1)
        vq2_stage_zq_buf = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)

        codebook = np.load(CODEBOOK_PATH).astype(np.float32)
        assert codebook.shape == (num_code, dim), f"codebook shape mismatch: {codebook.shape}"

        vq_codebook_buf = allocate(shape=(num_code, dim), dtype=np.float32, cacheable=1)
        vq_codebook_buf[:] = codebook
        vq_codebook_buf.sync_to_device()

        set_u64(vq_dequant.mmio, 0x1C, 0x20, vq_codebook_buf.device_address)
        write_float(vq_dequant.mmio, 0x34, current_dec_scale_inv)

        print("vq_dequant regs:", flush=True)
        print("  dec_scale_inv =", read_float(vq_dequant.mmio, 0x34), flush=True)

        print(f"Load {input_file}...", flush=True)
        idx_data = np.fromfile(input_file, dtype=np.uint16)
        if idx_data.shape != (num_vectors,):
            raise ValueError(f"Expected index shape ({num_vectors},), got {idx_data.shape}")
        print(f"index shape: {idx_data.shape}", flush=True)

        idx_min, idx_max = int(idx_data.min()), int(idx_data.max())
        print(f"idx range: [{idx_min}, {idx_max}]", flush=True)
        if idx_min < 0 or idx_max >= num_code:
            raise RuntimeError(f"Invalid index range: [{idx_min}, {idx_max}]")

        vq2_stage_idx_buf[:] = idx_data
        vq2_stage_idx_buf.sync_to_device()
        vq2_stage_zq_buf[:] = 0
        vq2_stage_zq_buf.sync_to_device()

        set_u64(vq_dequant.mmio, 0x10, 0x14, vq2_stage_idx_buf.device_address)
        set_u64(vq_dequant.mmio, 0x28, 0x2C, vq2_stage_zq_buf.device_address)

        print("Run vq_dequant...", flush=True)
        t0 = time.perf_counter()
        start_and_wait_old_style(vq_dequant.mmio, timeout_s=VQ_TIMEOUT_S)
        print(f"vq_dequant done: {(time.perf_counter() - t0) * 1000:.2f} ms", flush=True)

        vq2_stage_zq_buf.sync_from_device()
        zq_snapshot = np.array(vq2_stage_zq_buf, dtype=np.int8, copy=True)

        dec_in_np = np.ascontiguousarray(zq_snapshot.reshape(expected_dec_in))
        dec_out_np = np.empty(expected_dec_out, dtype=np.int8, order='C')

        print("Run Decoder...", flush=True)
        t0 = time.perf_counter()
        job_id = dec_runner.execute_async([dec_in_np], [dec_out_np])
        dec_runner.wait(job_id)
        print(f"Decoder done: {(time.perf_counter() - t0) * 1000:.2f} ms", flush=True)

        def _int8_region_stats(arr):
            x = arr[0].astype(np.float32)
            h, w, _ = x.shape
            regions = {
                'left1': x[:, :1, :],
                'left2': x[:, :2, :],
                'left4': x[:, :4, :],
                'left10': x[:, :10, :],
                'cols10_20': x[:, 10:20, :],
                'mid10': x[:, w//2-5:w//2+5, :],
                'right10': x[:, -10:, :],
            }
            out = {}
            for k, v in regions.items():
                rgb = v.mean(axis=(0, 1))
                out[k + '_int8_rgb_mean'] = [float(rgb[0]), float(rgb[1]), float(rgb[2])]
                out[k + '_int8_luma_mean'] = float((0.299*v[:,:,0] + 0.587*v[:,:,1] + 0.114*v[:,:,2]).mean())
            out['left10_minus_mid10_int8_rgb'] = [
                out['left10_int8_rgb_mean'][i] - out['mid10_int8_rgb_mean'][i]
                for i in range(3)
            ]
            out['left10_minus_mid10_int8_luma'] = out['left10_int8_luma_mean'] - out['mid10_int8_luma_mean']
            out['left1_minus_cols10_20_int8_luma'] = out['left1_int8_luma_mean'] - out['cols10_20_int8_luma_mean']
            out['left2_minus_cols10_20_int8_luma'] = out['left2_int8_luma_mean'] - out['cols10_20_int8_luma_mean']
            out['left4_minus_cols10_20_int8_luma'] = out['left4_int8_luma_mean'] - out['cols10_20_int8_luma_mean']
            out['global_min'] = int(x.min())
            out['global_max'] = int(x.max())
            return out

        int8_stats = _int8_region_stats(dec_out_np)
        stats_path = output_file + '.dec_out_int8_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as fp:
            json.dump(int8_stats, fp, ensure_ascii=False, indent=2)
        print('Decoder int8 stats ->', stats_path, flush=True)
        print(json.dumps(int8_stats, ensure_ascii=False), flush=True)

        post_lut = build_lut(current_dec_out_scale)
        recon_img = post_lut[dec_out_np[0].view(np.uint8)]

        output_dir = os.path.dirname(os.path.abspath(output_file))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        ok = cv2.imwrite(output_file, cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to save image: {output_file}")
        print(f"Image saved -> {output_file}", flush=True)

    finally:
        free_buf(vq2_stage_idx_buf, "vq2_stage_idx_buf")
        free_buf(vq2_stage_zq_buf, "vq2_stage_zq_buf")
        free_buf(vq_codebook_buf, "vq_codebook_buf")
        try:
            del dec_runner
        except Exception:
            pass


# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} encode/decode <input-file> <output-file>", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    if cmd == 'encode':
        cmd_encode(input_file, output_file)
    elif cmd == 'decode':
        cmd_decode(input_file, output_file)
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        print(f"usage: {sys.argv[0]} encode/decode <input-file> <output-file>", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
