import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPProcessor, CLIPModel


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    # _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    orig = posemb_grid.dtype
    posemb_grid = F.interpolate(posemb_grid.float(), size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.to(orig)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def initialize_clip(config, num_patches=240):
    clip_preprocessor = CLIPProcessor.from_pretrained(config['clip_name'])
    clip_model = CLIPModel.from_pretrained(config['clip_name'])

    num_patches = int(config['image_size']*config['image_size']/(16*16))
    pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())
    pos_embed.weight = resize_pos_embed(clip_model.visual.positional_embedding.unsqueeze(0), pos_embed.unsqueeze(0))
    clip_model.visual.positional_embedding = pos_embed
    return clip_model, clip_preprocessor