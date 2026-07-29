#!/usr/bin/env python3
"""Batch decoder for VCK190. XRT/VART resources are reused across indices."""

import argparse
import glob
import os

import cv2
import numpy as np

import cfg
from vck190_runtime import Vck190Runtime, decoder_i8_to_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default=cfg.IDX_DIR)
    parser.add_argument("--output-dir", default=cfg.RESULT_DIR)
    args = parser.parse_args()
    cfg.ensure_output_dirs()
    files = sorted(glob.glob(os.path.join(args.index_dir, "*.bin")))
    if not files:
        raise RuntimeError("No index files")
    os.makedirs(args.output_dir, exist_ok=True)
    runtime = Vck190Runtime()
    for path in files:
        index = np.fromfile(path, dtype=np.uint16).reshape(cfg.LATENT_H, cfg.LATENT_W)
        image = decoder_i8_to_bgr(runtime.decode(index), runtime.dec_out_scale)
        out = os.path.join(args.output_dir, os.path.splitext(os.path.basename(path))[0] + ".png")
        if not cv2.imwrite(out, image):
            raise RuntimeError(f"Cannot write {out}")
        print("decoded", path, "->", out)


if __name__ == "__main__":
    main()

