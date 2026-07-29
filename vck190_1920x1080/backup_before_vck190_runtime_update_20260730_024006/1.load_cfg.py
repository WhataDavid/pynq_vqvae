#!/usr/bin/env python3
"""Load and inspect the VCK190 DPU/VQ xclbin through XRT."""

import cfg


def main():
    cfg.require_versal_runtime()
    import pyxrt

    xbin = pyxrt.xclbin(cfg.XCLBIN_PATH)
    print("xclbin:", cfg.XCLBIN_PATH)
    print("kernels:", [kernel.get_name() for kernel in xbin.get_kernels()])
    device = pyxrt.device(cfg.DEVICE_INDEX)
    uuid = device.load_xclbin(xbin)
    print("loaded UUID:", uuid)


if __name__ == "__main__":
    main()

