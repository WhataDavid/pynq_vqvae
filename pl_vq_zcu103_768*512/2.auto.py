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

BIT_PATH      = os.path.join(WORK_DIR, 'pl_vq_zcu103_768*512/dpu.bit')
ENC_XMODEL    = os.path.join(WORK_DIR, 'xmodel/encoder_768x512.xmodel')
DEC_XMODEL    = os.path.join(WORK_DIR, 'xmodel/decoder_768x512.xmodel')
CODEBOOK_PATH = os.path.join(WORK_DIR, 'codebook.npy')

ENC_OUT_SCALE = 0.015625
DEC_IN_SCALE  = 0.03125
DEC_OUT_SCALE = 0.007812

# -----------------------------
# VQ AXI-Lite 寄存器（按 HLS 导出表）
# -----------------------------
REG_AP_CTRL        = 0x00

REG_IN_ADDR_L      = 0x10   # in_z_1
REG_IN_ADDR_H      = 0x14   # in_z_2

REG_CODEBOOK_L     = 0x1C   # in_codebook_1
REG_CODEBOOK_H     = 0x20   # in_codebook_2

REG_OUT_ADDR_L     = 0x28   # out_z_q_1
REG_OUT_ADDR_H     = 0x2C   # out_z_q_2

REG_ENC_SCALE      = 0x34   # enc_scale
REG_DEC_SCALE_INV  = 0x3C   # dec_scale_inv

# 关键：参照你给的稳定版本，启动用 0x11
VQ_START_CMD       = 0x11

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


def safe_get_attr(obj, key, default=None):
    try:
        if obj.has_attr(key):
            return obj.get_attr(key)
    except Exception:
        pass
    return default


def get_dpu_subgraph(xmodel_path):
    """
    查找第一个 device == 'DPU' 的 subgraph。
    注意：这里不能要求 is_leaf=True，因为你的 xmodel 里
    真正可执行的 DPU 子图是 non-leaf。
    """
    graph = xir.Graph.deserialize(xmodel_path)
    root = graph.get_root_subgraph()

    found = []

    def walk(sg):
        dev = safe_get_attr(sg, "device", None)
        if dev is not None and str(dev).upper() == "DPU":
            found.append(sg)

        try:
            if not sg.is_leaf:
                for c in sg.toposort_child_subgraph():
                    walk(c)
        except Exception:
            pass

    walk(root)

    if len(found) == 0:
        raise RuntimeError(f"No DPU subgraph found in xmodel: {xmodel_path}")

    if len(found) > 1:
        print(f"⚠️ Multiple DPU subgraphs found in {xmodel_path}, use first one:", flush=True)
        for i, sg in enumerate(found):
            try:
                print(f"  [{i}] {sg.get_name()}", flush=True)
            except Exception:
                print(f"  [{i}] <unknown>", flush=True)

    sg = found[0]
    print(f"✅ use DPU subgraph: {sg.get_name()}", flush=True)
    return sg


def to_hwc(arr):
    a = arr[0] if arr.ndim == 4 else arr
    if a.ndim != 3:
        raise RuntimeError(f"Bad tensor ndim: {arr.shape}")

    if a.shape[-1] in (1, 3):   # HWC
        return a
    if a.shape[0] in (1, 3):    # CHW
        return np.transpose(a, (1, 2, 0))

    raise RuntimeError(f"Bad tensor shape: {arr.shape}")


def postprocess_int8_to_u8(dec_out_int8):
    img = to_hwc(dec_out_int8).astype(np.float32)
    img = img * DEC_OUT_SCALE
    img = np.clip(img * 0.5 + 0.5, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


def save_rgb(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def print_arr_info(name, arr):
    print(
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"C={arr.flags['C_CONTIGUOUS']}, nbytes={arr.nbytes}",
        flush=True
    )


def print_tensor_info(prefix, tensor):
    print(f"{prefix}.name = {tensor.name}", flush=True)
    print(f"{prefix}.dims = {tuple(tensor.dims)}", flush=True)
    try:
        print(f"{prefix}.dtype = {tensor.dtype}", flush=True)
    except Exception:
        pass
    try:
        print(f"{prefix}.fix_point = {tensor.get_attr('fix_point')}", flush=True)
    except Exception:
        pass


def run_vq_once(vq_ip, vq_in, vq_out, cb_buf):
    """
    按你给出的稳定版本风格执行一次 VQ：
      - sync input/output
      - 重写地址
      - AP_CTRL 写 0x11 启动
      - 轮询 AP_DONE
      - sync output back
    """
    # 清零输出，避免 IP 未完整覆盖时留下旧值
    vq_out.fill(0)

    # 同步输入/输出到 device
    vq_in.sync_to_device()
    vq_out.sync_to_device()

    # 为稳妥起见，每次都重写固定寄存器
    vq_ip.mmio.write(REG_CODEBOOK_L, cb_buf.device_address & 0xFFFFFFFF)
    vq_ip.mmio.write(REG_CODEBOOK_H, (cb_buf.device_address >> 32) & 0xFFFFFFFF)
    vq_ip.mmio.write(REG_ENC_SCALE, f32_to_u32(ENC_OUT_SCALE))
    vq_ip.mmio.write(REG_DEC_SCALE_INV, f32_to_u32(1.0 / DEC_IN_SCALE))

    # 重写本次输入输出地址
    vq_ip.mmio.write(REG_IN_ADDR_L,  vq_in.device_address & 0xFFFFFFFF)
    vq_ip.mmio.write(REG_IN_ADDR_H,  (vq_in.device_address >> 32) & 0xFFFFFFFF)
    vq_ip.mmio.write(REG_OUT_ADDR_L, vq_out.device_address & 0xFFFFFFFF)
    vq_ip.mmio.write(REG_OUT_ADDR_H, (vq_out.device_address >> 32) & 0xFFFFFFFF)

    print(
        f"VQ addr in =0x{vq_in.device_address:016x}, "
        f"out=0x{vq_out.device_address:016x}",
        flush=True
    )

    ctrl_before = vq_ip.mmio.read(REG_AP_CTRL)
    print(f"VQ CTRL before start = 0x{ctrl_before:08x}", flush=True)

    # 关键：参照你给的版本，用 0x11 启动
    vq_ip.mmio.write(REG_AP_CTRL, VQ_START_CMD)

    print(f"⏳ VQ: start IP with 0x{VQ_START_CMD:02x}", flush=True)
    print("⏳ VQ: polling AP_DONE", flush=True)

    t0 = time.time()
    while True:
        ctrl = vq_ip.mmio.read(REG_AP_CTRL)
        if ctrl & 0x02:  # AP_DONE
            break
        if time.time() - t0 > 3.0:
            raise RuntimeError(f"VQ timeout, CTRL=0x{ctrl:08x}")
        time.sleep(0.001)

    print(f"✅ VQ done, CTRL=0x{ctrl:08x}", flush=True)

    # 输出同步回 CPU
    vq_out.sync_from_device()


def run():
    print("⏳ 连接 overlay...", flush=True)
    overlay = DpuOverlay(BIT_PATH, download=False)
    vq_ip = overlay.vq_accel_1
    print("✅ overlay connected", flush=True)

    print("⏳ 查找 encoder DPU subgraph...", flush=True)
    enc_sg = get_dpu_subgraph(ENC_XMODEL)

    print("⏳ 查找 decoder DPU subgraph...", flush=True)
    dec_sg = get_dpu_subgraph(DEC_XMODEL)

    print("⏳ 创建 DPU runner...", flush=True)
    enc_runner = vart.Runner.create_runner(enc_sg, "run")
    dec_runner = vart.Runner.create_runner(dec_sg, "run")
    print("✅ DPU runner ready", flush=True)

    enc_in_tensor  = enc_runner.get_input_tensors()[0]
    enc_out_tensor = enc_runner.get_output_tensors()[0]
    dec_in_tensor  = dec_runner.get_input_tensors()[0]
    dec_out_tensor = dec_runner.get_output_tensors()[0]

    print_tensor_info("ENC input ", enc_in_tensor)
    print_tensor_info("ENC output", enc_out_tensor)
    print_tensor_info("DEC input ", dec_in_tensor)
    print_tensor_info("DEC output", dec_out_tensor)

    enc_in_shape  = tuple(enc_in_tensor.dims)
    enc_out_shape = tuple(enc_out_tensor.dims)
    dec_in_shape  = tuple(dec_in_tensor.dims)
    dec_out_shape = tuple(dec_out_tensor.dims)

    print("ENC input shape :", enc_in_shape, flush=True)
    print("ENC output shape:", enc_out_shape, flush=True)
    print("DEC input shape :", dec_in_shape, flush=True)
    print("DEC output shape:", dec_out_shape, flush=True)

    assert enc_out_shape == dec_in_shape, "Encoder output / Decoder input mismatch"
    assert len(enc_out_shape) == 4, f"Unexpected latent shape: {enc_out_shape}"

    # 当前按 NHWC 解释
    _, h_lat, w_lat, c_lat = enc_out_shape
    num_vectors = h_lat * w_lat
    dim = c_lat

    print(
        f"latent: H={h_lat}, W={w_lat}, C={c_lat}, num_vectors={num_vectors}",
        flush=True
    )

    # -----------------------------
    # DPU buffer
    # -----------------------------
    print("⏳ allocate DPU buffers...", flush=True)
    enc_in_buf  = allocate(shape=enc_in_shape,  dtype=np.int8, cacheable=1)
    enc_out_buf = allocate(shape=enc_out_shape, dtype=np.int8, cacheable=1)
    dec_in_buf  = allocate(shape=dec_in_shape,  dtype=np.int8, cacheable=1)
    dec_out_buf = allocate(shape=dec_out_shape, dtype=np.int8, cacheable=1)
    print("✅ DPU buffers allocated", flush=True)

    print(f"enc_in_buf  addr = 0x{enc_in_buf.device_address:016x}", flush=True)
    print(f"enc_out_buf addr = 0x{enc_out_buf.device_address:016x}", flush=True)
    print(f"dec_in_buf  addr = 0x{dec_in_buf.device_address:016x}", flush=True)
    print(f"dec_out_buf addr = 0x{dec_out_buf.device_address:016x}", flush=True)

    # -----------------------------
    # VQ buffer
    # -----------------------------
    print("⏳ allocate VQ buffers...", flush=True)
    vq_in  = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)
    vq_out = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)
    print("✅ VQ buffers allocated", flush=True)

    print(f"vq_in  addr = 0x{vq_in.device_address:016x}", flush=True)
    print(f"vq_out addr = 0x{vq_out.device_address:016x}", flush=True)

    # -----------------------------
    # codebook
    # -----------------------------
    print(f"⏳ loading codebook: {CODEBOOK_PATH}", flush=True)
    if not os.path.exists(CODEBOOK_PATH):
        raise FileNotFoundError(f"codebook not found: {CODEBOOK_PATH}")

    codebook = np.load(CODEBOOK_PATH)
    print_arr_info("codebook(raw)", codebook)

    codebook = np.ascontiguousarray(codebook, dtype=np.float32)
    print_arr_info("codebook(f32)", codebook)

    assert codebook.ndim == 2, f"codebook ndim must be 2, got {codebook.ndim}"
    assert codebook.shape == (512, 64), \
        f"unexpected codebook shape: {codebook.shape}, expected (512, 64)"
    assert codebook.shape[1] == dim, \
        f"codebook dim mismatch: {codebook.shape}, dim={dim}"

    cb_buf = allocate(shape=codebook.shape, dtype=np.float32, cacheable=1)
    cb_buf[:] = codebook
    cb_buf.sync_to_device()

    print("✅ codebook buffer allocated", flush=True)
    print(f"cb_buf addr = 0x{cb_buf.device_address:016x}", flush=True)

    # -----------------------------
    # 初始固定寄存器配置
    # -----------------------------
    print("⏳ configure VQ IP registers...", flush=True)
    vq_ip.mmio.write(REG_CODEBOOK_L, cb_buf.device_address & 0xFFFFFFFF)
    vq_ip.mmio.write(REG_CODEBOOK_H, (cb_buf.device_address >> 32) & 0xFFFFFFFF)
    vq_ip.mmio.write(REG_ENC_SCALE, f32_to_u32(ENC_OUT_SCALE))
    vq_ip.mmio.write(REG_DEC_SCALE_INV, f32_to_u32(1.0 / DEC_IN_SCALE))

    print(
        f"VQ regs: codebook=0x{cb_buf.device_address:016x}, "
        f"enc_scale={ENC_OUT_SCALE}, dec_scale_inv={1.0 / DEC_IN_SCALE}",
        flush=True
    )

    files = sorted([f for f in os.listdir(PRE_DIR) if f.endswith('.npy')])
    print(f"📦 images: {len(files)}", flush=True)

    if len(files) == 0:
        print("⚠️ no input .npy found.", flush=True)
        return

    for i, fn in enumerate(files):
        print("\n" + "=" * 72, flush=True)
        print(f"[{i+1}/{len(files)}] file = {fn}", flush=True)
        print("=" * 72, flush=True)

        # -------------------------------------------------
        # 0) 读输入
        # -------------------------------------------------
        npy_path = os.path.join(PRE_DIR, fn)
        print(f"⏳ loading input: {npy_path}", flush=True)
        x = np.load(npy_path)
        print_arr_info("input(raw)", x)

        assert x.shape == (512, 768, 3), f"bad input shape: {x.shape}"
        assert x.dtype == np.int8, f"bad input dtype: {x.dtype}"

        x = np.ascontiguousarray(x, dtype=np.int8)
        print_arr_info("input(contig)", x)

        # -------------------------------------------------
        # 1) Encoder
        # -------------------------------------------------
        print("⏳ encoder: copy to CMA input", flush=True)
        enc_in_buf[0] = x
        enc_in_buf.sync_to_device()

        print("⏳ encoder: execute_async", flush=True)
        jid = enc_runner.execute_async([enc_in_buf], [enc_out_buf])

        print("⏳ encoder: wait", flush=True)
        enc_runner.wait(jid)
        enc_out_buf.sync_from_device()
        print("✅ encoder done", flush=True)

        enc_out = np.array(enc_out_buf, copy=True)
        print_arr_info("enc_out", enc_out)
        print(
            f"enc_out stat: min={enc_out.min()} max={enc_out.max()} "
            f"mean={enc_out.mean():.3f}",
            flush=True
        )

        # -------------------------------------------------
        # 2) VQ
        # -------------------------------------------------
        print("⏳ VQ: reshape encoder output", flush=True)
        src_flat = enc_out.reshape(num_vectors, dim)
        print_arr_info("src_flat", src_flat)

        print("⏳ VQ: copy to input buffer", flush=True)
        np.copyto(vq_in, src_flat)

        print("⏳ VQ: run once", flush=True)
        run_vq_once(vq_ip, vq_in, vq_out, cb_buf)

        zq_flat = np.array(vq_out, copy=True)
        print_arr_info("vq_out(flat)", zq_flat)
        print(
            f"vq_out stat: min={zq_flat.min()} max={zq_flat.max()} "
            f"mean={zq_flat.mean():.3f} nz={np.count_nonzero(zq_flat)}",
            flush=True
        )

        zq = zq_flat.reshape(dec_in_shape)
        print_arr_info("zq", zq)

        # -------------------------------------------------
        # 3) Decoder
        # -------------------------------------------------
        print("⏳ decoder: copy to CMA input", flush=True)
        dec_in_buf[:] = zq
        dec_in_buf.sync_to_device()

        print("⏳ decoder: execute_async", flush=True)
        jid = dec_runner.execute_async([dec_in_buf], [dec_out_buf])

        print("⏳ decoder: wait", flush=True)
        dec_runner.wait(jid)
        dec_out_buf.sync_from_device()
        print("✅ decoder done", flush=True)

        dec_out = np.array(dec_out_buf, copy=True)
        print_arr_info("dec_out", dec_out)
        print(
            f"dec_out stat: min={dec_out.min()} max={dec_out.max()} "
            f"mean={dec_out.mean():.3f}",
            flush=True
        )

        # -------------------------------------------------
        # 4) 后处理保存
        # -------------------------------------------------
        print("⏳ postprocess", flush=True)
        img = postprocess_int8_to_u8(dec_out)
        print_arr_info("img(u8)", img)

        out_png = os.path.join(RES_DIR, f"recon_{i}.png")
        save_rgb(out_png, img)
        print(f"✅ saved: {out_png}", flush=True)

    del enc_runner
    del dec_runner

    enc_in_buf.freebuffer()
    enc_out_buf.freebuffer()
    dec_in_buf.freebuffer()
    dec_out_buf.freebuffer()
    vq_in.freebuffer()
    vq_out.freebuffer()
    cb_buf.freebuffer()

    print("\n✅ done:", RES_DIR, flush=True)


if __name__ == "__main__":
    run()