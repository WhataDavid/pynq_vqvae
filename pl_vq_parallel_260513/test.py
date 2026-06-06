from pynq_dpu import DpuOverlay
overlay = DpuOverlay('/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae/pl_vq_parallel_260513/dpu.bit', download=False)
vq_ip = overlay.vq_accel_1

print("IP type:", type(vq_ip))
print("=== register_map ===")
print(vq_ip.register_map)

print("=== mmio len ===", vq_ip.mmio.length)
for off in [0x00,0x04,0x08,0x0c,0x10,0x14,0x18,0x1c,0x20,0x24,0x28,0x2c,0x30,0x34,0x38,0x3c]:
    try:
        print(hex(off), hex(vq_ip.mmio.read(off)))
    except Exception as e:
        print(hex(off), "ERR", e)