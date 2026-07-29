#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local environment and file-path configuration only.

When moving this project to another board/computer, edit this file for:
  - Python package search paths
  - project root path
  - bitstream / model / data / result paths

Algorithm parameters such as tensor shape, scale, queue depth, benchmark options,
and IP instance fallback names are intentionally kept inside each runtime script.
"""

import os
import sys

# ============================================================
# 1. Python environment paths
# ============================================================
# Keep these as None if your environment already exposes the packages.
SITE_PACKAGES_PATH = '/usr/lib/python3/site-packages'
DPU_PYNQ_PATH = '/home/xilinx/jupyter_notebooks/soft/DPU-PYNQ'


def setup_python_paths():
    """Add optional PYNQ/DPU Python paths before importing pynq_dpu/vart/xir."""
    if SITE_PACKAGES_PATH and SITE_PACKAGES_PATH not in sys.path:
        sys.path.append(SITE_PACKAGES_PATH)
    if DPU_PYNQ_PATH and DPU_PYNQ_PATH not in sys.path:
        sys.path.insert(0, DPU_PYNQ_PATH)


# ============================================================
# 2. Project / file paths
# ============================================================
WORK_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae'
PL_SUBDIR = 'zcu111_1280x720'
PL_DIR = os.path.join(WORK_DIR, PL_SUBDIR)
BIT_PATH = os.path.join(PL_DIR, 'dpu.bit')

PRE_DIR = os.path.join(WORK_DIR, 'imgs_preprocessed_1280x720')
CODEBOOK_PATH = os.path.join(WORK_DIR, 'codebook/codebook_som.npy')
VQ_CODEBOOK_PATH = os.path.join(WORK_DIR, 'codebook/codebook_som_vq_1280x720.npy')
ENC_XMODEL = os.path.join(WORK_DIR, 'xmodel/encoder_1280x720_som.xmodel')
DEC_XMODEL = os.path.join(WORK_DIR, 'xmodel/decoder_1280x720_som.xmodel')
# CODEBOOK_PATH = os.path.join(WORK_DIR, 'codebook.npy')
# ENC_XMODEL = os.path.join(WORK_DIR, 'xmodel/encoder_1280x720.xmodel')
# DEC_XMODEL = os.path.join(WORK_DIR, 'xmodel/decoder_1280x720.xmodel')

RES_DIR = os.path.join(PL_DIR, 'results_1280x720')
IDX_DIR = os.path.join(RES_DIR, 'idx_bins')


def ensure_output_dirs():
    """Create output directories used by the runtime scripts."""
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(IDX_DIR, exist_ok=True)
