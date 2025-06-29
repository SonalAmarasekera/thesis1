# models/separator_rwkv6.py
import torch.nn as nn
from dac.nn.layers import Snake1d as Snake
#from snake_act import Snake
#from rwkv_block.rwkv6_wrapper import RWKV6Wrapper    # wrapper above
# models/separator_rwkv6.py  (only the __init__ signature & conv lines change)

class RWKV6Separator(nn.Module):
    def __init__(self, codec, depth=4, n_spk=2, down_ratio=2):
        super().__init__()
        self.codec = codec.eval().requires_grad_(False)

        # 1️⃣  Detect latent channels directly from the codec
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 16000, device=next(codec.parameters()).device)
            latent_dim = codec.encode(dummy)[0].shape[1]        # 1024 for 16 kHz model

        hidden_dim = latent_dim // down_ratio                  # e.g. 512

        # 2️⃣  Build adapters with the detected size
        self.down = nn.Conv1d(latent_dim, hidden_dim, 1)
        self.rwkv = RWKV6Wrapper(depth=depth, dim=hidden_dim)  # wrapper from Day-3 memo
        self.up   = nn.Conv1d(hidden_dim, latent_dim, 1)

        from dac.nn.layers import Snake1d as Snake              # correct Snake import
        self.head = nn.ModuleList([
            nn.Sequential(nn.Conv1d(latent_dim, latent_dim, 1),
                          Snake(latent_dim))
            for _ in range(n_spk)
        ])


    def forward(self, mix_wave, state=None):
        mix_wave = mix_wave.to(self.down.weight.dtype)   # force dtype = model
        z = self.codec.encode(mix_wave)[0]          # [B,128,F]
        y = self.down(z)
        y, state = self.rwkv(y, state)
        y = self.up(y)
        latents = [h(y) for h in self.head]
        return latents, state
