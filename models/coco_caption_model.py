import torch
from torch import nn
import torch.nn.functional as F

from models.coco_caption_utils import initialize_clip
from models.coco_caption_layers import TransformerDecoder, MemoryAdapterLayer


class CocoCaptioner(nn.Module):
    def __init__(self, tokenizer = None, word_embedder = None, config = None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer 
        self.word_embedder = word_embedder
        self.positional_embedding = nn.Embedding(config['max_words_per_cap'], 768)

        self.visual_encoder, _ = initialize_clip(config)
        self.text_decoder = TransformerDecoder(
            config['num_layers'], 
            768, 
            config['nhead'], 
            config['dim_feedforward'], 
            config['dropout']
        )
        self.memory_adapter = MemoryAdapterLayer(dim_query=768, dim_mem=512)
        self.fc = nn.Linear(768, config['vocab_size'])

    def _cal_loss(self, prediction, caption_ids):
        """Calculate the loss between prediction and caption.
        
        Args:
            prediction (tensor): (batch_size, max_words_per_cap, vocab_size)
                For each word in caption, the model will predict a logit of next word.
            caption (tensor): (batch_size, max_words_per_cap)
                The ground truth caption. For each word in caption, it is the index of in vocabulary.
        
        Returns:
            loss (tensor): (1)
        """
        caption_ids = caption_ids.masked_fill(caption_ids == self.tokenizer.pad_token_id, -100)
        prediction = prediction.view(-1, prediction.size(2))
        caption_ids = caption_ids.view(-1)
        loss = F.cross_entropy(prediction, caption_ids, ignore_index=-100, reduction='mean')
        return loss
            
    def forward(self, image, caption=None, is_train=True):
        # image: (batch_size, 3, 224, 224)
        image = image.to(dtype=next(self.parameters()).dtype) 
        # image_embeds: (batch_size, 512)
        image_embeds = self.visual_encoder.get_image_features(image)
        image_embeds = image_embeds.unsqueeze(1) # (batch_size, 1, 512)

        if is_train:      
            # caption_embeds: (batch_size, max_words_per_cap, 768)
            caption_embeds = self.word_embedder(caption.input_ids, attention_mask=caption.attention_mask).last_hidden_state 
            position_ids = torch.arange(0, caption.input_ids.size(1)).unsqueeze(0)
            positional_embeddings = self.positional_embedding(position_ids)
            caption_embeds = caption_embeds + positional_embeddings

            # caption_embeds, image_embeds = self.memory_adapter(caption_embeds, image_embeds)
            image_embeds = nn.Linear(512, 768)(image_embeds) # (batch_size, 1, 768)

            seq_len = caption_embeds.size(1)
            self_attn_mask = torch.ones((seq_len, seq_len), device=self.config['device']).bool()
            self_attn_mask = torch.triu(self_attn_mask, diagonal=1).bool()
            # output: (batch_size, max_words_per_cap, 768)
            output, self.self_attn_w, self.cross_attn_w = self.text_decoder(caption_embeds, image_embeds, self_attn_mask)
            print(output)
            # output: (batch_size, max_words_per_cap, vocab_size)
            output = self.fc(output)
            
            prediction = F.softmax(output, dim=2)
            loss = self._cal_loss(prediction, caption.input_ids)
            print(loss)
            return loss
        return image_embeds