#!/usr/bin/env python3
"""
只需执行一次：烧写 bitfile 到 FPGA
之后只要不断电、不重新烧写，bit 一直有效
"""
import os, struct
import numpy as np
from pynq import allocate
from pynq_dpu import DpuOverlay

WORK_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae'

print("⏳ 正在加载 bitfile，请稍候...")
overlay = DpuOverlay(os.path.join(WORK_DIR, 'pl_vq_zcu103_768*512/dpu.bit'))
print("✅ Bitfile 加载完成，FPGA 已编程。")
print("   现在可以直接运行 2.auto.py，无需重新加载 bit。")