# config_local.py
# -*- coding: utf-8 -*-

import os

# ============================================================
# Basic project paths
# ============================================================
WORK_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae'

PL_DIR = os.path.join(WORK_DIR, 'zcu111_index_dequant')

PRE_DIR = os.path.join(WORK_DIR, 'imgs_preprocessed')

CODEBOOK_PATH = os.path.join(WORK_DIR, 'codebook.npy')

ENC_XMODEL = os.path.join(WORK_DIR, 'xmodel/encoder_768x512.xmodel')
DEC_XMODEL = os.path.join(WORK_DIR, 'xmodel/decoder_768x512.xmodel')

BIT_PATH = os.path.join(PL_DIR, 'dpu_debug.bit')

RES_DIR = os.path.join(PL_DIR, 'results_768x512_serial_cv2')
IDX_DIR = os.path.join(RES_DIR, 'idx_bins')


# ============================================================
# Python import paths
# ============================================================
PYTHON_SITE_PACKAGES = '/usr/lib/python3/site-packages'
DPU_PYNQ_PATH = '/home/xilinx/jupyter_notebooks/soft/DPU-PYNQ'


# ============================================================
# Model / image shape
# ============================================================
IMG_H = 512
IMG_W = 768

LATENT_H = 128
LATENT_W = 192

DIM = 64
NUM_CODE = 512


# ============================================================
# Quantization scales
# ============================================================
ENC_OUT_SCALE = 0.015625
DEC_IN_SCALE = 0.03125
DEC_OUT_SCALE = 0.007812

DEC_SCALE_INV = 1.0 / DEC_IN_SCALE


# ============================================================
# Overlay behavior
# ============================================================
# 如果你先运行 1.load.py 加载 bitstream，这里用 False
# 如果希望脚本自己下载 bitstream，这里用 True
DOWNLOAD_BITSTREAM = False


# ============================================================
# IP names
# ============================================================
VQ_ACCEL_IP = 'vq_accel_1'
VQ_DEQUANT_IP = 'vq_dequant_1'