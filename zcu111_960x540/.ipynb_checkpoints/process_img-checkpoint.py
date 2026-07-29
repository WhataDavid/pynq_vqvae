import os
import numpy as np
from PIL import Image as PILImage

# --- 配置 ---
IMG_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae/imgs'

NPY_SAVE_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae/imgs_preprocessed_960x540'
JPG_SAVE_DIR = '/home/xilinx/jupyter_notebooks/duxu/pynq_vqvae/imgs_preprocessed_960x540'

TARGET_SIZE = (960, 540)  # (W, H)

# 这里的 scale 需要和 DPU Encoder 输入 scale 一致: 2^-6 = 0.015625
ENC_IN_SCALE = 0.015625

os.makedirs(NPY_SAVE_DIR, exist_ok=True)
os.makedirs(JPG_SAVE_DIR, exist_ok=True)

files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

print(f"开始预处理 {len(files)} 张图片...")

for i, f in enumerate(files):
    img_path = os.path.join(IMG_DIR, f)

    # 1. 加载与缩放
    img_pil = PILImage.open(img_path).convert('RGB')
    img_small = img_pil.resize(TARGET_SIZE, PILImage.BILINEAR)

    # 2. 归一化到 [-1, 1]
    img_arr = np.asarray(img_small, dtype=np.float32)
    input_fp32 = (img_arr / 255.0 - 0.5) / 0.5

    # 3. 量化为 int8，作为 DPU Encoder 输入
    input_int8 = np.clip(
        np.round(input_fp32 / ENC_IN_SCALE),
        -128,
        127
    ).astype(np.int8)

    # 4. 保存 int8 .npy
    npy_save_path = os.path.join(NPY_SAVE_DIR, f + '.npy')
    np.save(npy_save_path, input_int8)

    # 5. 将 int8 输入反量化回 float，用于保存可视化 jpg
    #    int8 -> fp32 [-1,1] 近似
    input_dequant_fp32 = input_int8.astype(np.float32) * ENC_IN_SCALE

    #    [-1,1] -> [0,255]
    input_dequant_uint8 = np.clip(
        (input_dequant_fp32 * 0.5 + 0.5) * 255.0,
        0,
        255
    ).round().astype(np.uint8)

    # 6. 保存量化后可视化 JPG
    base_name = os.path.splitext(f)[0]
    jpg_save_path = os.path.join(JPG_SAVE_DIR, base_name + '_int8_input.jpg')

    PILImage.fromarray(input_dequant_uint8, mode='RGB').save(
        jpg_save_path,
        quality=95
    )

    print(f"[{i+1}/{len(files)}] {f}")
    print(f"  npy: {npy_save_path}")
    print(f"  jpg: {jpg_save_path}")
    print(f"  int8 range: [{int(input_int8.min())}, {int(input_int8.max())}]")

print("✅ 处理完成")
print(f"NPY 数据保存在: {NPY_SAVE_DIR}")
print(f"int8 可视化 JPG 保存在: {JPG_SAVE_DIR}")