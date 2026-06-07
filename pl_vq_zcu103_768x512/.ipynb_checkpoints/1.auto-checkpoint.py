#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

WORK_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae'
BIT_PATH = os.path.join(WORK_DIR, 'pl_vq_zcu103_768x512/dpu.bit')

sys.path.append('/usr/lib/python3/site-packages')
sys.path.insert(0, '/home/xilinx/jupyter_notebooks/soft/DPU-PYNQ')

from pynq_dpu import DpuOverlay


def main():
    print("⏳ 正在加载 bitstream，请稍候...", flush=True)
    overlay = DpuOverlay(BIT_PATH, download=True)
    print("✅ Bitstream 加载完成，FPGA 已编程。", flush=True)
    print("   现在可以直接运行 2.auto.py，无需重新加载 bit。", flush=True)

    # 防止局部变量太早释放（通常不是必须，但保留无妨）
    _ = overlay


if __name__ == "__main__":
    main()