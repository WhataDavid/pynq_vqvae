#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

WORK_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae'
BIT_PATH = os.path.join(WORK_DIR, 'zcu103_index_dequant/dpu.bit')

sys.path.append('/usr/lib/python3/site-packages')
sys.path.insert(0, '/home/xilinx/jupyter_notebooks/soft/DPU-PYNQ')

from pynq_dpu import DpuOverlay


def main():
    print("Loading bitstream...", flush=True)
    overlay = DpuOverlay(BIT_PATH, download=True)
    print("Bitstream loaded.", flush=True)
    _ = overlay


if __name__ == "__main__":
    main()