# VCK190 deployment staging directory

This directory contains a VCK190/Versal XRT and VART deployment, not a ZCU111
PYNQ Overlay deployment.  It must be copied to a VCK190 board running the
matching Vitis-AI 2022.1/PetaLinux image with `pyxrt`, `vart`, and `xir`.

Load `vck190_dpu_vq.xclbin` through `1.load_cfg.py`.  Do not try to load the
VCK190 PDI or xclbin with `pynq_dpu.DpuOverlay` on a ZCU111.

The VQ search IP uses `codebook_som_dpu_search_1920x1080_vck190.npy`.  The
dequant IP uses `codebook_som.npy`; the two files must not be swapped.
