from torch import nn

from models.coco_caption_utils import initialize_clip


class CocoCaptioner(nn.Module):
    def __init__(self, tokenizer = None, config = None):
        super().__init__()
        self.tokenizer = tokenizer 
        self.visual_encoder, _ = initialize_clip(config)
        self.text_decoder = None  
        self.beam_generator = None
            
    def forward(self, image, caption=None, is_train=True):
        pass