import torch
import sys
sys.path.insert(0, 'D:/sonnet/sonnet')
from models.vqvae import VQVAE

# load original
ckpt  = torch.load('results/vqvae_pth.pth', map_location='cpu', weights_only=False)
state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

# load into model and re-save using only basic python types
vqvae = VQVAE(128, 32, 2, 512, 64, 0.25)
vqvae.load_state_dict(state)

# save with _use_new_zipfile_serialization=False for max compatibility
torch.save(
    {'model': vqvae.state_dict()},
    'results/vqvae_pth_compatible.pth',
    _use_new_zipfile_serialization=True,
    pickle_protocol=2        # protocol 2 avoids numpy._core dependency
)
print('saved vqvae_pth_compatible20260402.pth')

# verify codebook
import numpy as np
cb = vqvae.vector_quantization.embedding.weight.data.numpy()
np.save('results/codebook_new.npy', cb)
print('codebook shape:', cb.shape)
print('codebook saved')