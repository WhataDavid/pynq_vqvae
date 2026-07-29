#!/usr/bin/env python3
"""Paths and runtime checks for the VCK190 1920x1080 VQ-VAE deployment."""

import os

WORK_DIR = "/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae"
PL_DIR = os.path.join(WORK_DIR, "vck190_1920x1080")
RESULT_DIR = os.path.join(PL_DIR, "results_1920x1080")
IDX_DIR = os.path.join(RESULT_DIR, "idx_bins")

XCLBIN_PATH = os.path.join(PL_DIR, "vck190_dpu_vq.xclbin")
PDI_PATH = os.path.join(PL_DIR, "vck190_dpu_vq.pdi")
XSA_PATH = os.path.join(PL_DIR, "vck190_dpu_vq.xsa")
HWH_PATH = os.path.join(PL_DIR, "vck190_dpu_vq.hwh")

ENC_XMODEL = os.path.join(PL_DIR, "encoder_vck190_1920x1080.xmodel")
DEC_XMODEL = os.path.join(PL_DIR, "decoder_vck190_1920x1080.xmodel")
CODEBOOK_PATH = os.path.join(PL_DIR, "codebook_som.npy")
VQ_CODEBOOK_PATH = os.path.join(PL_DIR, "codebook_som_dpu_search_1920x1080_vck190.npy")

PRE_DIR = os.path.join(WORK_DIR, "imgs_preprocessed_1920x1080")
DEVICE_INDEX = int(os.environ.get("VCK190_DEVICE_INDEX", "0"))

IMG_W, IMG_H = 1920, 1080
LATENT_W, LATENT_H = 480, 270
DIM, NUM_CODE = 64, 512
NUM_Z = LATENT_W * LATENT_H


def ensure_output_dirs():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(IDX_DIR, exist_ok=True)


def require_versal_runtime():
    """Fail early on a ZCU111/PYNQ image instead of loading a wrong image."""
    try:
        import pyxrt  # noqa: F401
        import vart  # noqa: F401
        import xir  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "VCK190 requires the Vitis-AI/PetaLinux XRT runtime (pyxrt, vart, xir). "
            "The ZCU111 PYNQ image is only a staging location and cannot run this design."
        ) from exc

