#!/usr/bin/env python3
"""Single-image VCK190 VQ-VAE encode/decode test using XRT and VART."""

import argparse
import os
import time

import cv2
import numpy as np

import cfg
from vck190_runtime import Vck190Runtime, decoder_i8_to_bgr, preprocess_image


def cmd_encode(runtime, image_path, index_path):
    t0 = time.perf_counter()
    encoded = preprocess_image(image_path, runtime.enc_in_scale)
    index, _ = runtime.encode(encoded)
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    index.astype(np.uint16).tofile(index_path)
    print(f"encode: {index_path}, {index.size} indices, {(time.perf_counter() - t0) * 1000:.2f} ms")


def cmd_decode(runtime, index_path, output_path):
    t0 = time.perf_counter()
    index = np.fromfile(index_path, dtype=np.uint16)
    index = index.reshape(cfg.LATENT_H, cfg.LATENT_W)
    output = runtime.decode(index)
    image = decoder_i8_to_bgr(output, runtime.dec_out_scale)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not cv2.imwrite(output_path, image):
        raise RuntimeError(f"Cannot write {output_path}")
    print(f"decode: {output_path}, {(time.perf_counter() - t0) * 1000:.2f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("encode", "decode", "roundtrip"))
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    cfg.ensure_output_dirs()
    runtime = Vck190Runtime()
    print("enc scales:", runtime.enc_in_scale, runtime.enc_out_scale)
    print("dec scales:", runtime.dec_in_scale, runtime.dec_out_scale)
    if args.command == "encode":
        cmd_encode(runtime, args.input, args.output)
    elif args.command == "decode":
        cmd_decode(runtime, args.input, args.output)
    else:
        index_path = os.path.join(cfg.IDX_DIR, "roundtrip.bin")
        cmd_encode(runtime, args.input, index_path)
        cmd_decode(runtime, index_path, args.output)


if __name__ == "__main__":
    main()

