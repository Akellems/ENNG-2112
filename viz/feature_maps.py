
# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
# File: viz/feature_maps.py
# Purpose: Dump intermediate feature maps into grid images for diagnosis.
# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤

from typing import Dict
import torch
from PIL import Image
import numpy as np

@torch.no_grad()
def dump_feature_grid(model, x, layer, max_channels=16) -> Image.Image:
    model.eval(); feats={}
    def hook(m,i,o): feats['fm']=o.detach().cpu()
    h = layer.register_forward_hook(hook)
    _ = model(x)
    h.remove()
    fm = feats['fm'][0]  # [C,H,W]
    C = min(max_channels, fm.size(0))
    fm = fm[:C]
    fm = (fm - fm.min(dim=(1,2), keepdim=True)[0]) / (fm.max(dim=(1,2), keepdim=True)[0] + 1e-9)
    fm = (fm*255).byte().numpy()  # [C,H,W]
    tiles = [Image.fromarray(ch) for ch in fm]
    # make a grid (4x4 by default for 16 channels)
    grid_side = int(np.ceil(C**0.5))
    H,W = tiles[0].size
    canvas = Image.new('L', (grid_side*W, grid_side*H))
    for idx,t in enumerate(tiles):
        r,c = divmod(idx, grid_side)
        canvas.paste(t, (c*W, r*H))
    return canvas.convert('RGB')
