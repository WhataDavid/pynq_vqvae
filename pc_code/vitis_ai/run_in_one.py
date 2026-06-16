"""
VQ-VAE 量化脚本（修复版）
输入尺寸：700×500，与训练时一致
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, '/workspace/prj_without_onnx')
from models.encoder import Encoder
from models.decoder import Decoder
from models.vqvae   import VQVAE
from pytorch_nndct.apis import torch_quantizer

# ══════════════════════════════════════════════════════
# 0. 超参数
# ══════════════════════════════════════════════════════
N_HIDDENS          = 128
N_RESIDUAL_HIDDENS = 32
N_RESIDUAL_LAYERS  = 2
EMBEDDING_DIM      = 64
N_EMBEDDINGS       = 512
BETA               = 0.25

PTH_PATH      = '/workspace/prj_without_onnx/vqvae/results/vqvae_compatible.pth'
CALIB_JPG_DIR = '/workspace/prj_without_onnx/vqvae/data/raw_images'
CODEBOOK_PATH = '/workspace/prj_without_onnx/vqvae/results/codebook.npy'
OUTPUT_DIR    = '/workspace/prj_without_onnx/vqvae/results'

# ✅ 与训练时一致
IMG_H, IMG_W = 512, 768    # transforms.Resize((700, 500)) → H=500, W=700

DEVICE = torch.device('cpu')

# ══════════════════════════════════════════════════════
# 1. Wrapper
# ══════════════════════════════════════════════════════
class EncoderWrapper(nn.Module):
    def __init__(self, vqvae):
        super().__init__()
        self.encoder               = vqvae.encoder
        self.pre_quantization_conv = vqvae.pre_quantization_conv
    def forward(self, x):
        return self.pre_quantization_conv(self.encoder(x))

class DecoderWrapper(nn.Module):
    def __init__(self, vqvae):
        super().__init__()
        self.decoder = vqvae.decoder
    def forward(self, zq):
        return self.decoder(zq)

# ══════════════════════════════════════════════════════
# 2. 加载模型
# ══════════════════════════════════════════════════════
def load_vqvae():
    model = VQVAE(N_HIDDENS, N_RESIDUAL_HIDDENS, N_RESIDUAL_LAYERS,
                  N_EMBEDDINGS, EMBEDDING_DIM, BETA)
    ckpt  = torch.load(PTH_PATH, map_location=DEVICE, weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"✅ 模型加载成功")
    return model

# ══════════════════════════════════════════════════════
# 3. 校准数据
# ══════════════════════════════════════════════════════
_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),        # ✅ 500×700，与训练一致
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

def get_encoder_calib_data(n=100):
    exts  = ('.jpg','.jpeg','.png','.bmp')
    files = sorted([
        os.path.join(CALIB_JPG_DIR, f)
        for f in os.listdir(CALIB_JPG_DIR)
        if f.lower().endswith(exts)
    ])[:n]
    if not files:
        raise FileNotFoundError(f"未在 {CALIB_JPG_DIR} 找到图片")
    data = []
    for path in files:
        img    = Image.open(path).convert('RGB')
        tensor = _transform(img).unsqueeze(0)    # (1,3,500,700)
        data.append(tensor)
    print(f"  ✅ Encoder 校准：{len(data)} 张，shape={data[0].shape}")
    return data

def get_decoder_calib_data(n=100):
    np.random.seed(42)
    codebook = np.load(CODEBOOK_PATH).astype(np.float32)
    H_lat = IMG_H // 4   # 125
    W_lat = IMG_W // 4   # 175
    data = []
    for _ in range(n):
        idx = np.random.randint(0, codebook.shape[0], size=(H_lat, W_lat))
        zq  = codebook[idx].transpose(2,0,1)[np.newaxis]  # (1,64,125,175)
        data.append(torch.from_numpy(zq))
    print(f"  ✅ Decoder 校准：{len(data)} 个，shape={data[0].shape}")
    return data

# ══════════════════════════════════════════════════════
# 4. 量化 + 导出
# ════════════════════════════════════════════════════��═
def quantize_and_export(model, dummy_input, calib_data, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*55}\n  量化: {name}  input={tuple(dummy_input.shape)}\n{'='*55}")

    # 校准
    print(f"  [1/3] 校准（{len(calib_data)} 样本）...")
    quantizer   = torch_quantizer('calib', model, dummy_input,
                                  output_dir=output_dir, device=DEVICE)
    quant_model = quantizer.quant_model
    quant_model.eval()
    with torch.no_grad():
        for i, x in enumerate(calib_data):
            quant_model(x)
            if (i+1) % 20 == 0:
                print(f"    {i+1}/{len(calib_data)}")
    quantizer.export_quant_config()
    print("  ✅ 校准完成")

    # 测试
    print("  [2/3] 测试...")
    quantizer   = torch_quantizer('test', model, dummy_input,
                                  output_dir=output_dir, device=DEVICE)
    quant_model = quantizer.quant_model
    quant_model.eval()
    with torch.no_grad():
        out = quant_model(dummy_input)
    print(f"  ✅ 输出形状: {tuple(out.shape)}")

    # 导出
    print("  [3/3] 导出 xmodel...")
    quantizer.export_xmodel(output_dir=output_dir, deploy_check=True)
    print(f"  ✅ 导出完成")

# ══════════════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════════════
def main():
    print(f"输入尺寸: {IMG_H}×{IMG_W} (H×W)")

    vqvae     = load_vqvae()
    enc_model = EncoderWrapper(vqvae).eval()
    dec_model = DecoderWrapper(vqvae).eval()

    # dummy input
    dummy_enc = torch.randn(1, 3,  IMG_H,      IMG_W)       # (1,3,500,700)
    dummy_dec = torch.randn(1, 64, IMG_H//4,   IMG_W//4)    # (1,64,125,175)
    print(f"Encoder dummy: {tuple(dummy_enc.shape)}")
    print(f"Decoder dummy: {tuple(dummy_dec.shape)}")

    # 验证 forward
    with torch.no_grad():
        ze = enc_model(dummy_enc)
        xh = dec_model(dummy_dec)
    print(f"Encoder 输出: {tuple(ze.shape)}")   # (1,64,125,175)
    print(f"Decoder 输出: {tuple(xh.shape)}")   # (1,3,500,700)

    enc_calib = get_encoder_calib_data(n=100)
    dec_calib = get_decoder_calib_data(n=100)

    quantize_and_export(enc_model, dummy_enc, enc_calib, OUTPUT_DIR, 'EncoderWrapper')
    quantize_and_export(dec_model, dummy_dec, dec_calib, OUTPUT_DIR, 'DecoderWrapper')

    print(f"""
{'='*55}
🎉 量化完成！执行以下命令编译 xmodel：
{'='*55}

  vai_c_xir \\
    -x {OUTPUT_DIR}/EncoderWrapper_int.xmodel \\
    -a /tmp/zcu111_arch.json \\
    -o {OUTPUT_DIR}/compiled_encoder_768x512 \\
    -n encoder_zcu111

  vai_c_xir \\
    -x {OUTPUT_DIR}/DecoderWrapper_int.xmodel \\
    -a /tmp/zcu111_arch.json \\
    -o {OUTPUT_DIR}/compiled_decoder_768x512 \\
    -n decoder_zcu111
""")

if __name__ == '__main__':
    main()