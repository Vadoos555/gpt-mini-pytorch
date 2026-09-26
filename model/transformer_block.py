import torch
import torch.nn as nn

from model.multi_head_attention import MultiHeadAttention
from model.mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        
        x = x + self.mlp(self.norm2(x))
        
        return x
    

if __name__ == "__main__":
    torch.manual_seed(42)
    block = TransformerBlock(d_model=256, num_heads=4)

    x = torch.randn(2, 3, 256)
    y = block(x)

    print("x:", x.shape)
    print("y:", y.shape)
    