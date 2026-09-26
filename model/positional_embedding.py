import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, context_size, d_model):
        super().__init__()
        
        self.embedding = nn.Embedding(context_size, d_model)
    
    def forward(self, positions):
        return self.embedding(positions)


if __name__ == '__main__':
    context_size = 256
    d_model = 256

    embedding = PositionalEmbedding(context_size, d_model)

    positions = torch.tensor([0, 1, 2])
    vectors = embedding(positions)

    print("positions: ", positions)
    print()
    
    print("vectors shape: ", vectors.shape)
    print()
    
    print("embedding matrix shape: ", embedding.embedding.weight.shape)
    