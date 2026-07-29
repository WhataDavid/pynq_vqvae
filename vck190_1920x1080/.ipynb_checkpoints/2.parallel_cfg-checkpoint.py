#!/usr/bin/env python3
"""Batch encoder for VCK190. XRT/VART resources are reused across frames."""

import argparse
import glob
import os

import cfg
from vck190_runtime import Vck190Runtime, preprocess_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--pattern", default="*.jpg")
    args = parser.parse_args()
    cfg.ensure_output_dirs()
    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise RuntimeError("No input images")
    runtime = Vck190Runtime()
    for path in files:
        index, _ = runtime.encode(preprocess_image(path, runtime.enc_in_scale))
        name = os.path.splitext(os.path.basename(path))[0] + ".bin"
        out = os.path.join(cfg.IDX_DIR, name)
        index.tofile(out)
        print("encoded", path, "->", out)


if __name__ == "__main__":
    main()

