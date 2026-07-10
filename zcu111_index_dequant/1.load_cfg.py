#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cfg

cfg.setup_python_paths()

from pynq_dpu import DpuOverlay


def main():
    print("Loading bitstream...", flush=True)
    print("BIT_PATH:", cfg.BIT_PATH, flush=True)
    overlay = DpuOverlay(cfg.BIT_PATH, download=True)
    print("Bitstream loaded.", flush=True)
    _ = overlay


if __name__ == "__main__":
    main()
