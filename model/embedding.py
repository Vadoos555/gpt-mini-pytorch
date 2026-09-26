import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, ids):
        return self.embedding(ids)


if __name__ == '__main__':
    embedding = torch.nn.Embedding(10_000, 256)

    ids = torch.tensor([9, 3])

    vectors = embedding(ids)

    print(vectors.shape)
    print(embedding.weight.shape)

    print()
    print(torch.equal(vectors[0], embedding.weight[9]))
    print(torch.equal(vectors[1], embedding.weight[3]))
    