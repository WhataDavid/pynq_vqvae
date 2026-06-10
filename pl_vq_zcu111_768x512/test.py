#!/usr/bin/env python3
"""VQVAE DPU pipeline: encode or decode a single file.

Usage:
    encode <input-npy> <output-bin>
    decode <input-bin> <output-png>
"""
import sys, time, struct
import numpy as np
import cv2

# sys.path = [p for p in sys.path if '/usr/local' not in p]
# sys.path.insert(0, '/usr/local/python3.10/lib/python3.10')
# sys.path.insert(0, '/usr/local/python3.10/lib/python3.10/lib-dynload')
# sys.path.append('/usr/local/python3.10/lib/python3.10/site-packages')
WORK_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae'
sys.path.append('/usr/lib/python3/site-packages')
sys.path.insert(0, '/home/xilinx/jupyter_notebooks/soft/DPU-PYNQ')


from pynq_dpu import DpuOverlay
import vart, xir
from pynq import allocate

ROOT = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae'

BIT_PATH    = ROOT + '/pl_vq_zcu111_768x512/dpu.bit'
ENC_XMODEL  = ROOT + '/xmodel/encoder_768x512.xmodel'
DEC_XMODEL  = ROOT + '/xmodel/decoder_768x512.xmodel'
CODEBOOK    = ROOT + '/codebook.npy'

enc_out_scale = 0.015625
dec_in_scale  = 0.03125
dec_out_scale = 0.007812
num_vectors = 128 * 192
dim = 64

# ==================== Load Bitstream ====================
print('Loading bitstream...', flush=True)
overlay = DpuOverlay(BIT_PATH, download=True)
vq_ip = overlay.vq_accel_1
print('Bitstream loaded.', flush=True)

# ==================== Load DPU Runners ====================
def get_dpu_subgraph(path):
    graph = xir.Graph.deserialize(path)
    return graph, graph.get_root_subgraph().toposort_child_subgraph()[1]

enc_graph, enc_subgraph = get_dpu_subgraph(ENC_XMODEL)
enc_runner = vart.Runner.create_runner(enc_subgraph, 'run')
dec_graph, dec_subgraph = get_dpu_subgraph(DEC_XMODEL)
dec_runner = vart.Runner.create_runner(dec_subgraph, 'run')

# ==================== Init VQ IP ====================
vq_in_buf = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)
vq_out_buf = allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1)

vq_in_cb = allocate(shape=(512, 64), dtype=np.float32, cacheable=1)
vq_in_cb[:] = np.load(CODEBOOK).astype(np.float32)
vq_in_cb.sync_to_device()

vq_ip.register_map.in_codebook_1 = vq_in_cb.device_address & 0xFFFFFFFF
vq_ip.register_map.in_codebook_2 = vq_in_cb.device_address >> 32
vq_ip.register_map.enc_scale = struct.unpack('<I', struct.pack('<f', enc_out_scale))[0]
vq_ip.register_map.dec_scale_inv = struct.unpack('<I', struct.pack('<f', 1.0 / dec_in_scale))[0]

# ==================== Post LUT ====================
post_lut = np.zeros(256, dtype=np.uint8)
for i in range(256):
    val_int8 = np.int8(i)
    val_fp32 = float(val_int8) * dec_out_scale
    val_norm = max(0.0, min(1.0, val_fp32 * 0.5 + 0.5))
    post_lut[i] = int(val_norm * 255.0)

# ==================== CLI ====================
def encode(input_path, output_path):
    input_data = np.load(input_path)

    in_buf = [np.ascontiguousarray(input_data[np.newaxis])]
    out_buf = [np.ndarray((1, 128, 192, 64), dtype=np.int8, buffer=vq_in_buf.data)]
    job_id = enc_runner.execute_async(in_buf, out_buf)
    enc_runner.wait(job_id)

    vq_in_buf.sync_to_device()
    vq_ip.mmio.write(0x10, vq_in_buf.device_address & 0xFFFFFFFF)
    vq_ip.mmio.write(0x14, vq_in_buf.device_address >> 32)
    vq_ip.mmio.write(0x28, vq_out_buf.device_address & 0xFFFFFFFF)
    vq_ip.mmio.write(0x2C, vq_out_buf.device_address >> 32)
    vq_ip.mmio.write(0x00, 0x11)
    while (vq_ip.mmio.read(0x00) & 0x02) == 0:
        time.sleep(0.001)
    vq_out_buf.sync_from_device()

    zq = np.array(vq_out_buf, copy=True)
    with open(output_path, 'wb') as f:
        f.write(zq.tobytes())
    print(f'Encoded: {input_path} -> {output_path}', flush=True)

def decode(input_path, output_path):
    with open(input_path, 'rb') as f:
        zq = np.frombuffer(f.read(), dtype=np.int8).reshape(1, 128, 192, 64).copy()

    dec_in_buf = [zq]
    dec_out_buf = [np.empty((1, 512, 768, 3), dtype=np.int8, order='C')]
    job_id = dec_runner.execute_async(dec_in_buf, dec_out_buf)
    dec_runner.wait(job_id)

    recon_img = post_lut[dec_out_buf[0][0].view(np.uint8)]
    out_png = output_path if output_path.endswith('.png') else output_path + '.png'
    cv2.imwrite(out_png, cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR))
    print(f'Decoded: {input_path} -> {out_png}', flush=True)

if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] not in ('encode', 'decode'):
        print(f'Usage: {sys.argv[0]} encode|decode <input-file> <output-file>', flush=True)
        sys.exit(1)

    cmd, input_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

    if cmd == 'encode':
        encode(input_file, output_file)
    else:
        decode(input_file, output_file)

    del enc_runner, dec_runner
