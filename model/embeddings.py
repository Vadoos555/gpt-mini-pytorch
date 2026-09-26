import torch
import torch.nn as nn

from model.embedding import TokenEmbedding
from model.positional_embedding import PositionalEmbedding


class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, context_size, d_model):
        super().__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(context_size, d_model)
    
    def forward(self, ids):
        token_vectors = self.token_embedding(ids)
        
        positions = torch.arange(ids.size(1), device=ids.device)
        position_vectors = self.position_embedding(positions)
        
        x = token_vectors + position_vectors
        
        return x


if __name__ == '__main__':
    vocab_size = 10_000
    context_size = 256
    d_model = 256

    embedding = GPTEmbedding(
        vocab_size,
        context_size,
        d_model
    )

    ids = torch.tensor([9, 3])

    x = embedding(ids)

    print("ids: ", ids)
    print()
    
    print("x shape: ", x.shape)
    