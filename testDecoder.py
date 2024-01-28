import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, Dropout, Linear

class DecoderLayer(nn.Module):
    def __init__(self, units, num_heads=1, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiheadAttention(embed_dim=units, num_heads=num_heads, dropout=dropout_rate)
        self.cross_attention = MultiheadAttention(embed_dim=units, num_heads=num_heads, dropout=dropout_rate)

        # Feedforward
        self.ff = nn.Sequential(
            Linear(units, units * 4),
            nn.ReLU(),
            Dropout(dropout_rate),
            Linear(units * 4, units)
        )

        self.dropout = Dropout(dropout_rate)

    def forward(self, image_features, partial_caption):
        # Self-attention
        out_seq2, _ = self.self_attention(out_seq, out_seq, out_seq)
        out_seq = out_seq + self.dropout(out_seq2)

        # Cross-attention
        out_seq2, self.last_attention_scores = self.cross_attention(out_seq, in_seq, in_seq)
        out_seq = out_seq + self.dropout(out_seq2)

        # Feedforward
        out_seq = self.ff(out_seq)

        return out_seq
    
if __name__ == "__main__":
    embedding_size = 512
    num_heads = 8

    decoder_layer = DecoderLayer(units=embedding_size, num_heads=num_heads)

    image_features = torch.rand(1, embedding_size) 
    image_features = image_features.unsqueeze(0)

    partial_caption = torch.rand(30, 1, embedding_size) 
    output = decoder_layer(image_features, partial_caption)

    print(output.shape)