import torch
import torch.nn as nn

from model.embeddings import GPTEmbedding
from model.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embedding = GPTEmbedding(config.vocab_size, config.context_size, config.d_model)
        self.blocks = nn.ModuleList([ 
                    TransformerBlock(config.d_model, config.num_heads)
                     for _ in range(config.num_layers)
            ])
        
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        
    def forward(self, ids):
        x = self.embedding(ids)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


if __name__ == '__main__':
    from config.model_config import GPTConfig
    
    config = GPTConfig()
    model = GPT(config)
    total = sum(
        p.numel() for p in model.parameters()
    )
    print(f'parameters: {total / 1_000_000: .2f}M')
    
    for name, p in model.named_parameters():
        print(name, p.shape, p.numel())
    
    ids = torch.randint(0, config.vocab_size, (2, 3))
    logits = model(ids)
    
    print('ids:', ids.shape)
    print('logits:', logits.shape)
    