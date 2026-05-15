import os, sys, time, threading, queue, struct
import numpy as np
import cv2
from pynq import allocate

# --- 1. 环境初始化 ---
WORK_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae'
sys.path.append('/usr/lib/python3/site-packages')
sys.path.insert(0, '/home/xilinx/jupyter_notebooks/soft/DPU-PYNQ')

from pynq_dpu import DpuOverlay
import vart, xir

print("loading bitstream")
overlay = DpuOverlay(os.path.join(WORK_DIR, 'pl_vq_parallel_260513/dpu.bit'))
print("loading complete")
vq_ip = overlay.vq_accel_1

enc_out_scale = 0.015625 
dec_in_scale  = 0.03125  
dec_out_scale = 0.007812 
num_vectors = 125 * 175
dim = 64

# --- 2. 极致性能配置 ---
num_bufs = 3
vq_in_bufs = [allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1) for _ in range(num_bufs)]
vq_out_bufs = [allocate(shape=(num_vectors, dim), dtype=np.int8, cacheable=1) for _ in range(num_bufs)]

vq_in_cb = allocate(shape=(512, 64), dtype=np.float32, cacheable=1)
vq_in_cb[:] = np.load(os.path.join(WORK_DIR, 'codebook.npy')).astype(np.float32)
vq_in_cb.sync_to_device() 

vq_ip.register_map.in_codebook_1 = vq_in_cb.device_address & 0xFFFFFFFF
vq_ip.register_map.in_codebook_2 = vq_in_cb.device_address >> 32
vq_ip.register_map.enc_scale = struct.unpack('<I', struct.pack('<f', enc_out_scale))[0]
vq_ip.register_map.dec_scale_inv = struct.unpack('<I', struct.pack('<f', 1.0 / dec_in_scale))[0]

def get_dpu_subgraph(path):
    graph = xir.Graph.deserialize(path)
    return graph, graph.get_root_subgraph().toposort_child_subgraph()[1] 

enc_graph, enc_subgraph = get_dpu_subgraph(os.path.join(WORK_DIR, 'xmodel/encoder_zcu111_700x500_old.xmodel'))
enc_runner = vart.Runner.create_runner(enc_subgraph, "run")
dec_graph, dec_subgraph = get_dpu_subgraph(os.path.join(WORK_DIR, 'xmodel/decoder_zcu111_700x500_old.xmodel'))
dec_runner = vart.Runner.create_runner(dec_subgraph, "run")

PRE_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae/imgs_preprocessed'
data_files = sorted([f for f in os.listdir(PRE_DIR) if f.endswith('.npy')])
num_imgs = len(data_files)

RES_DIR = './results'
if not os.path.exists(RES_DIR): os.makedirs(RES_DIR)

# --- 预计算 Decoder 输出的 LUT ---
post_lut = np.zeros(256, dtype=np.uint8)
for i in range(256):
    val_int8 = np.int8(i) 
    val_fp32 = float(val_int8) * dec_out_scale 
    val_norm = max(0.0, min(1.0, val_fp32 * 0.5 + 0.5)) 
    post_lut[i] = int(val_norm * 255.0)

# ==============================================================================
# ==================== Phase 1: 读取 I/O + Encoder + VQ ====================
# ==============================================================================
read_queue = queue.Queue(maxsize=3)
free_queue = queue.Queue(maxsize=num_bufs)
enc_res_queue = queue.Queue(maxsize=num_bufs)
for i in range(num_bufs):
    free_queue.put(i)

all_zq_results = []

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
        target_vq_buf = vq_in_bufs[buf_idx]
        out_buf = [np.ndarray((1, 125, 175, 64), dtype=np.int8, buffer=target_vq_buf.data)]
        
        job_id = enc_runner.execute_async(in_buf, out_buf)
        enc_runner.wait(job_id)
        
        enc_res_queue.put(buf_idx)

def phase1_pipeline():
    print(f" [Phase 1] Starting Read -> Encoder -> VQ Pipeline: {num_imgs} images")
    t_read = threading.Thread(target=read_worker)
    t_enc = threading.Thread(target=enc_worker)
    
    start_time = time.time()
    t_read.start()
    t_enc.start()

    while True:
        buf_idx = enc_res_queue.get()
        if buf_idx is None: 
            break
            
        curr_in_buf = vq_in_bufs[buf_idx]
        curr_out_buf = vq_out_bufs[buf_idx]
        
        curr_in_buf.sync_to_device()  
        
        vq_ip.mmio.write(0x10, curr_in_buf.device_address & 0xFFFFFFFF)
        vq_ip.mmio.write(0x14, curr_in_buf.device_address >> 32)
        vq_ip.mmio.write(0x28, curr_out_buf.device_address & 0xFFFFFFFF)
        vq_ip.mmio.write(0x2C, curr_out_buf.device_address >> 32)
        
        vq_ip.mmio.write(0x00, 0x11) 
        while (vq_ip.mmio.read(0x00) & 0x02) == 0:
            time.sleep(0.001) 
            
        curr_out_buf.sync_from_device() 
        
        zq_snapshot = np.array(curr_out_buf, copy=True)
        all_zq_results.append(zq_snapshot)
        
        free_queue.put(buf_idx)
        
    t_read.join()
    t_enc.join()
    phase1_time = time.time() - start_time
    print(f" [Phase 1] Finished! Includes NPY Reading. Time: {phase1_time*1000:.2f} ms | FPS: {num_imgs/phase1_time:.2f}\n")


# ==============================================================================
# ==================== Phase 2: Decoder + LUT (完全去写盘I/O) ================
# ==============================================================================
dec_out_queue = queue.Queue(maxsize=3)
all_recon_imgs = []

def dec_worker():
    for zq_res in all_zq_results:
        z_q_int8 = zq_res.reshape(1, 125, 175, 64)
        dec_in_buf = [z_q_int8]
        # 必须给每一帧分配独立空间，防止被并行处理的 LUT 线程覆盖
        dec_out_buf = [np.empty((1, 500, 700, 3), dtype=np.int8, order='C')]
        
        job_id = dec_runner.execute_async(dec_in_buf, dec_out_buf)
        dec_runner.wait(job_id)
        
        dec_out_queue.put(dec_out_buf[0])
        
    dec_out_queue.put(None)

def lut_worker():
    while True:
        dec_data = dec_out_queue.get()
        if dec_data is None:
            break
        
        # 极速后处理，并将结果存入内存列表
        recon_img = post_lut[dec_data[0].view(np.uint8)]
        all_recon_imgs.append(recon_img)

def phase2_pipeline():
    print(f" [Phase 2] Starting Decoder -> LUT Pipeline: {num_imgs} images")
    t_dec = threading.Thread(target=dec_worker)
    t_lut = threading.Thread(target=lut_worker)
    
    start_time = time.time()
    t_dec.start()
    t_lut.start()

    t_dec.join()
    t_lut.join()
    
    phase2_time = time.time() - start_time
    print(f" [Phase 2] Finished! (Excludes Image Saving). Time: {phase2_time*1000:.2f} ms | FPS: {num_imgs/phase2_time:.2f}\n")


# ==============================================================================
# ==================== 主控与收尾 ====================
# ==============================================================================
if __name__ == "__main__":
    # 执行包含 Read 的 Phase 1
    phase1_pipeline()
    
    # 执行 Decoder + LUT 并行的 Phase 2
    phase2_pipeline()
    
    # 释放 Runner
    del enc_runner
    del dec_runner
    print(" DPU Runners released.")
    
    # 统一单线程写盘
    print(" 正在写入最终图片...")
    for i, img in enumerate(all_recon_imgs):
        cv2.imwrite(f'{RES_DIR}/recon_{i}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
    print(f" Pipeline perfectly completed. Saved {len(all_recon_imgs)} images.")