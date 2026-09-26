from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 10_003
    context_size: int = 256
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 6
    dropout: float = 0.1
    
    
    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError('d_model must be divisible by num_heads')


if __name__ == '__main__':
    config = GPTConfig()
    
    print(config)
    print('head dimension:', config.d_model // config.num_heads)
    