#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import xir

WORK_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae'
ENC_XMODEL = os.path.join(WORK_DIR, 'xmodel/encoder_768x512.xmodel')
DEC_XMODEL = os.path.join(WORK_DIR, 'xmodel/decoder_768x512.xmodel')


def safe_get_attr(obj, key):
    try:
        if obj.has_attr(key):
            return obj.get_attr(key)
    except Exception:
        pass
    return None


def dump_subgraph(sg, indent=0):
    pad = "  " * indent

    try:
        name = sg.get_name()
    except Exception:
        name = "<no-name>"

    try:
        is_leaf = sg.is_leaf
    except Exception:
        is_leaf = "<unknown>"

    device = safe_get_attr(sg, "device")
    runner = safe_get_attr(sg, "runner")
    sg_type = safe_get_attr(sg, "type")
    core_id = safe_get_attr(sg, "device_core_id")
    core = safe_get_attr(sg, "device_core")
    arch = safe_get_attr(sg, "dpu_fingerprint")

    print(f"{pad}name   : {name}")
    print(f"{pad}is_leaf: {is_leaf}")
    print(f"{pad}device : {device}")
    print(f"{pad}runner : {runner}")
    print(f"{pad}type   : {sg_type}")
    print(f"{pad}core_id: {core_id}")
    print(f"{pad}core   : {core}")
    print(f"{pad}finger : {arch}")
    print(f"{pad}" + "-" * 50)

    if not is_leaf:
        try:
            children = sg.toposort_child_subgraph()
        except Exception as e:
            print(f"{pad}child error: {e}")
            return

        for c in children:
            dump_subgraph(c, indent + 1)


def inspect_one(path):
    print("=" * 80)
    print("XMODEL:", path)
    print("=" * 80)
    g = xir.Graph.deserialize(path)
    root = g.get_root_subgraph()
    dump_subgraph(root)
    print("")


def main():
    inspect_one(ENC_XMODEL)
    inspect_one(DEC_XMODEL)


if __name__ == "__main__":
    main()