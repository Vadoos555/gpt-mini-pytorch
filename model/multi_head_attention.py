import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        if d_model % num_heads != 0:
            raise ValueError('d_model must be divisible by num_heads')
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.shape
            
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
            
        q = q.view(B, T, self.num_heads, self.d_head)
        k = k.view(B, T, self.num_heads, self.d_head)
        v = v.view(B, T, self.num_heads, self.d_head)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.d_head)
        
        mask = torch.tril(torch.ones(T, T, device=x.device))
        
        scores = scores.masked_fill(mask==0, float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)
        
        y = weights @ v
        y = y.transpose(1, 2)
        y = y.contiguous().view(B, T, C)
        y = self.out(y)
        
        return y


if __name__ == "__main__":
    torch.manual_seed(42)

    attention = MultiHeadAttention(
        d_model=256,
        num_heads=4
    )

    x = torch.randn(2, 3, 256)

    y = attention(x)

    print("x:", x.shape)
    print("y:", y.shape)
    