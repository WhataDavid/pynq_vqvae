import os, sys, time, threading, queue, struct
import numpy as np
import cv2
from pynq import allocate

# --- 1. 环境初始化 ---
WORK_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae/pl_vq_zcu111_vq_index/'
sys.path.append('/usr/lib/python3/site-packages')
sys.path.insert(0, '/home/xilinx/jupyter_notebooks/soft/DPU-PYNQ')

from pynq_dpu import DpuOverlay
import vart, xir

overlay = DpuOverlay('/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae/pl_vq_zcu111_vq_index/dpu.bit')
vq_ip = overlay.vq_accel_1
print(vq_ip)

# 缩放系数（与HLS、原业务保持一致）
enc_out_scale = 0.015625
dec_in_scale  = 0.03125
dec_out_scale = 0.007812
dec_scale_inv = 1.0 / dec_in_scale

# 维度配置
num_vectors = 128 * 192
dim = 64
num_code = 512

# --- 2. 加载码本（Python反量化查表使用）---
codebook = np.load('/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae/codebook.npy').astype(np.float32)
assert codebook.shape == (num_code, dim), "码本维度错误"

# --- 3. Buffer 配置 ---
num_bufs = 3
# vq_accel 输入：Encoder输出 int8 特征
vq_in_bufs = [allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1) for _ in range(num_bufs)]
# vq_accel 输出：码本索引 uint16
vq_idx_bufs = [allocate(shape=(num_vectors,), dtype=np.uint16, cacheable=1) for _ in range(num_bufs)]

# 硬件端码本（vq_accel 仍需要）
vq_in_cb = allocate(shape=(num_code, dim), dtype=np.float32, cacheable=1)
vq_in_cb[:] = codebook
vq_in_cb.sync_to_device()

# ===================== 重点修正：寄存器地址严格对齐HLS报表 =====================
# in_codebook_1 = 0x1C , in_codebook_2 = 0x20
vq_ip.mmio.write(0x1C, vq_in_cb.device_address & 0xFFFFFFFF)
vq_ip.mmio.write(0x20, vq_in_cb.device_address >> 32)

# enc_scale = 0x34
vq_ip.mmio.write(0x34, struct.unpack('<I', struct.pack('<f', enc_out_scale))[0])
# dec_scale_inv = 0x3C
vq_ip.mmio.write(0x3C, struct.unpack('<I', struct.pack('<f', dec_scale_inv))[0])

# --- DPU 模型加载 ---
def get_dpu_subgraph(path):
    graph = xir.Graph.deserialize(path)
    return graph, graph.get_root_subgraph().toposort_child_subgraph()[1]

enc_graph, enc_subgraph = get_dpu_subgraph('/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae/xmodel/encoder_768x512.xmodel')
enc_runner = vart.Runner.create_runner(enc_subgraph, "run")
dec_graph, dec_subgraph = get_dpu_subgraph('/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae/xmodel/decoder_768x512.xmodel')
dec_runner = vart.Runner.create_runner(dec_subgraph, "run")

# --- 数据集路径 ---
PRE_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae/imgs_preprocessed'
data_files = sorted([f for f in os.listdir(PRE_DIR) if f.endswith('.npy')])
num_imgs = len(data_files)

RES_DIR = './results_768x512'
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

# --- Decoder 输出后处理LUT ---
post_lut = np.zeros(256, dtype=np.uint8)
for i in range(256):
    val_int8 = np.int8(i)
    val_fp32 = float(val_int8) * dec_out_scale
    val_norm = max(0.0, min(1.0, val_fp32 * 0.5 + 0.5))
    post_lut[i] = int(val_norm * 255.0)

# --- Python 软件反量化函数：索引 -> 特征 ---
def dequantize(index_arr: np.ndarray) -> np.ndarray:
    """
    软件反量化
    :param index_arr: 码本索引 (num_vectors,)
    :return: 反量化后特征 (num_vectors, dim) int8
    """
    # 索引查表
    feat_fp = codebook[index_arr]
    # 缩放 + 截断转int8，和硬件逻辑对齐
    feat_fp = feat_fp * dec_scale_inv
    feat_int8 = np.clip(feat_fp, -128, 127).astype(np.int8)
    return feat_int8

# ==============================================================================
# Phase 1: 读数据 -> Encoder -> vq_accel(输出索引) -> Python反量化
# ==============================================================================
read_queue = queue.Queue(maxsize=3)
free_queue = queue.Queue(maxsize=num_bufs)
enc_res_queue = queue.Queue(maxsize=num_bufs)
dequant_queue = queue.Queue(maxsize=num_bufs)

for i in range(num_bufs):
    free_queue.put(i)

# 存放反量化后、送入Decoder的特征
all_dequant_results = []

def read_worker():
    for f in data_files:
        data = np.load(os.path.join(PRE_DIR, f))
        read_queue.put(data)
    read_queue.put(None)

def enc_worker():
    while True:
        input_data = read_queue.get()
        if input_data is None:
            enc_res_queue.put(None)
            break

        in_buf = [np.ascontiguousarray(input_data[np.newaxis])]
        buf_idx = free_queue.get()
        target_in_buf = vq_in_bufs[buf_idx]

        out_buf = [np.ndarray((1, 128, 192, 64), dtype=np.int8, buffer=target_in_buf.data)]
        job_id = enc_runner.execute_async(in_buf, out_buf)
        enc_runner.wait(job_id)

        enc_res_queue.put(buf_idx)

def vq_worker():
    while True:
        buf_idx = enc_res_queue.get()
        if buf_idx is None:
            dequant_queue.put(None)
            break

        curr_in_buf = vq_in_bufs[buf_idx]
        curr_idx_buf = vq_idx_bufs[buf_idx]

        # 数据下发到硬件
        curr_in_buf.sync_to_device()

        # 输入特征 in_z 地址：0x10 / 0x14 (寄存器表一致，保留)
        vq_ip.mmio.write(0x10, curr_in_buf.device_address & 0xFFFFFFFF)
        vq_ip.mmio.write(0x14, curr_in_buf.device_address >> 32)
        # 输出索引 out_idx 地址：0x28 / 0x2C (寄存器表一致，保留)
        vq_ip.mmio.write(0x28, curr_idx_buf.device_address & 0xFFFFFFFF)
        vq_ip.mmio.write(0x2C, curr_idx_buf.device_address >> 32)

        # 启动IP并等待完成
        vq_ip.mmio.write(0x00, 0x11)
        while (vq_ip.mmio.read(0x00) & 0x02) == 0:
            time.sleep(0.001)

        # 取回索引数据
        curr_idx_buf.sync_from_device()
        idx_data = np.array(curr_idx_buf, copy=True)

        # 可选：打印极值，调试用（正常范围 0~511）
        print(f"Index range: min={idx_data.min()}, max={idx_data.max()}")

        # Python 软件反量化
        dequant_feat = dequantize(idx_data)
        all_dequant_results.append(dequant_feat)

        free_queue.put(buf_idx)

def phase1_pipeline():
    print(f"🚀 [Phase 1] Read -> Encoder -> VQ-Accel -> Python Dequant: {num_imgs} images")
    t_read = threading.Thread(target=read_worker)
    t_enc = threading.Thread(target=enc_worker)
    t_vq = threading.Thread(target=vq_worker)

    start_time = time.time()
    t_read.start()
    t_enc.start()
    t_vq.start()

    t_read.join()
    t_enc.join()
    t_vq.join()

    phase1_time = time.time() - start_time
    print(f"✅ [Phase 1] Finished! Time: {phase1_time*1000:.2f} ms | FPS: {num_imgs/phase1_time:.2f}\n")

# ==============================================================================
# Phase 2: Decoder 推理 + 图像后处理
# ==============================================================================
dec_out_queue = queue.Queue(maxsize=3)
all_recon_imgs = []

def dec_worker():
    for feat in all_dequant_results:
        dec_in = feat.reshape(1, 128, 192, 64)
        dec_in_buf = [dec_in]
        dec_out_buf = [np.empty((1, 512, 768, 3), dtype=np.int8, order='C')]

        job_id = dec_runner.execute_async(dec_in_buf, dec_out_buf)
        dec_runner.wait(job_id)
        dec_out_queue.put(dec_out_buf[0])
    dec_out_queue.put(None)

def lut_worker():
    while True:
        dec_data = dec_out_queue.get()
        if dec_data is None:
            break
        recon_img = post_lut[dec_data[0].view(np.uint8)]
        all_recon_imgs.append(recon_img)

def phase2_pipeline():
    print(f"🚀 [Phase 2] Decoder -> LUT Postprocess: {num_imgs} images")
    t_dec = threading.Thread(target=dec_worker)
    t_lut = threading.Thread(target=lut_worker)

    start_time = time.time()
    t_dec.start()
    t_lut.start()

    t_dec.join()
    t_lut.join()

    phase2_time = time.time() - start_time
    print(f"✅ [Phase 2] Finished. Time: {phase2_time*1000:.2f} ms | FPS: {num_imgs/phase2_time:.2f}\n")

# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    phase1_pipeline()
    phase2_pipeline()

    del enc_runner
    del dec_runner
    print("🧹 DPU Runners released.")

    print("🖼️ 正在写入最终图片...")
    for i, img in enumerate(all_recon_imgs):
        cv2.imwrite(f'{RES_DIR}/recon_{i}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"✅ Pipeline completed. Saved {len(all_recon_imgs)} images.")