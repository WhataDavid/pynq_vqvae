#!/usr/bin/env python3
"""VCK190 XRT/VART runtime shared by the 1920x1080 VQ-VAE scripts."""

import os
import time

import cv2
import numpy as np

import cfg


def _scale(tensor):
    try:
        return 2.0 ** (-tensor.get_attr("fix_point"))
    except Exception:
        return 1.0


def _dpu_subgraph(xir, xmodel_path):
    graph = xir.Graph.deserialize(xmodel_path)
    children = graph.get_root_subgraph().toposort_child_subgraph()
    dpu = [node for node in children if node.has_attr("device") and node.get_attr("device").upper() == "DPU"]
    if len(dpu) != 1:
        raise RuntimeError(f"Expected one DPU subgraph in {xmodel_path}, got {len(dpu)}")
    return dpu[0]


def _kernel_name(xclbin, base, instance):
    available = [kernel.get_name() for kernel in xclbin.get_kernels()]
    if base not in available:
        raise RuntimeError(f"Kernel {base!r} absent from xclbin; found {available}")
    return f"{base}:{{{instance}}}"


class Vck190Runtime:
    """Owns the XRT device/xclbin, VQ kernels, and VART DPU runners."""

    def __init__(self):
        cfg.require_versal_runtime()
        import pyxrt
        import vart
        import xir

        self.xrt = pyxrt
        self.device = pyxrt.device(cfg.DEVICE_INDEX)
        self.xclbin = pyxrt.xclbin(cfg.XCLBIN_PATH)
        self.uuid = self.device.load_xclbin(self.xclbin)
        self.vq_accel = pyxrt.kernel(
            self.device, self.uuid, _kernel_name(self.xclbin, "vq_accel", "vq_accel_1")
        )
        self.vq_dequant = pyxrt.kernel(
            self.device, self.uuid, _kernel_name(self.xclbin, "vq_dequant", "vq_dequant_1")
        )

        self.enc_runner = vart.Runner.create_runner(_dpu_subgraph(xir, cfg.ENC_XMODEL), "run")
        self.dec_runner = vart.Runner.create_runner(_dpu_subgraph(xir, cfg.DEC_XMODEL), "run")
        self.enc_in_scale = _scale(self.enc_runner.get_input_tensors()[0])
        self.enc_out_scale = _scale(self.enc_runner.get_output_tensors()[0])
        self.dec_in_scale = _scale(self.dec_runner.get_input_tensors()[0])
        self.dec_out_scale = _scale(self.dec_runner.get_output_tensors()[0])

        self.search_codebook = np.load(cfg.VQ_CODEBOOK_PATH).astype(np.float32, copy=False)
        self.decoder_codebook = np.load(cfg.CODEBOOK_PATH).astype(np.float32, copy=False)
        if self.search_codebook.shape != (cfg.NUM_CODE, cfg.DIM):
            raise ValueError(f"Bad VQ search codebook shape: {self.search_codebook.shape}")
        if self.decoder_codebook.shape != (cfg.NUM_CODE, cfg.DIM):
            raise ValueError(f"Bad decoder codebook shape: {self.decoder_codebook.shape}")

        self._search_cb_bo = self._bo(self.vq_accel, 1, self.search_codebook.nbytes)
        self._write_bo(self._search_cb_bo, self.search_codebook)
        self._decoder_cb_bo = self._bo(self.vq_dequant, 1, self.decoder_codebook.nbytes)
        self._write_bo(self._decoder_cb_bo, self.decoder_codebook)

    def _bo(self, kernel, arg_index, nbytes):
        return self.xrt.bo(self.device, nbytes, self.xrt.bo.normal, kernel.group_id(arg_index))

    def _write_bo(self, bo, array):
        raw = np.ascontiguousarray(array).view(np.uint8)
        bo.write(raw, 0)
        bo.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, raw.nbytes, 0)

    def _read_bo(self, bo, dtype, shape):
        raw = bo.read(np.dtype(dtype).itemsize * int(np.prod(shape)), 0)
        return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)

    def encode(self, input_nhwc_i8):
        expected = (1, cfg.IMG_H, cfg.IMG_W, 3)
        if input_nhwc_i8.shape != expected:
            raise ValueError(f"Encoder input {input_nhwc_i8.shape}, expected {expected}")
        ze = np.empty((1, cfg.LATENT_H, cfg.LATENT_W, cfg.DIM), dtype=np.int8)
        job = self.enc_runner.execute_async([np.ascontiguousarray(input_nhwc_i8)], [ze])
        self.enc_runner.wait(job)

        in_bo = self._bo(self.vq_accel, 0, ze.nbytes)
        idx_bo = self._bo(self.vq_accel, 2, cfg.NUM_Z * np.dtype(np.uint16).itemsize)
        self._write_bo(in_bo, ze.reshape(cfg.NUM_Z, cfg.DIM))
        run = self.vq_accel(in_bo, self._search_cb_bo, idx_bo, float(self.enc_out_scale), float(1.0 / self.dec_in_scale))
        run.wait()
        idx = self._read_bo(idx_bo, np.uint16, (cfg.NUM_Z,))
        return idx.reshape(cfg.LATENT_H, cfg.LATENT_W), ze

    def decode(self, index_hw):
        index_hw = np.ascontiguousarray(index_hw, dtype=np.uint16)
        if index_hw.shape != (cfg.LATENT_H, cfg.LATENT_W):
            raise ValueError(f"Index shape {index_hw.shape}, expected {(cfg.LATENT_H, cfg.LATENT_W)}")
        idx_bo = self._bo(self.vq_dequant, 0, index_hw.nbytes)
        zq_bo = self._bo(self.vq_dequant, 2, cfg.NUM_Z * cfg.DIM)
        self._write_bo(idx_bo, index_hw.reshape(cfg.NUM_Z))
        run = self.vq_dequant(idx_bo, self._decoder_cb_bo, zq_bo, float(1.0 / self.dec_in_scale))
        run.wait()
        zq = self._read_bo(zq_bo, np.int8, (1, cfg.LATENT_H, cfg.LATENT_W, cfg.DIM))
        out = np.empty((1, cfg.IMG_H, cfg.IMG_W, 3), dtype=np.int8)
        job = self.dec_runner.execute_async([zq], [out])
        self.dec_runner.wait(job)
        return out


def preprocess_image(path, enc_scale):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (cfg.IMG_W, cfg.IMG_H), interpolation=cv2.INTER_LINEAR)
    normalized = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.clip(np.round(normalized / enc_scale), -128, 127).astype(np.int8)[None, ...]


def decoder_i8_to_bgr(output, dec_scale):
    rgb = output[0].astype(np.float32) * dec_scale
    rgb = np.clip((rgb * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def elapsed_ms(start):
    return (time.perf_counter() - start) * 1000.0

