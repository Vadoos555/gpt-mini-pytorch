import torch

from config.model_config import GPTConfig
from model.gpt import GPT


def test_gpt_full_model():
    torch.manual_seed(42)
    
    config = GPTConfig()
    model = GPT(config)
    
    ids = torch.randint(0, config.vocab_size, (2, 3))
    logits = model(ids)
    
    # input shape
    assert ids.shape == (2, 3)
    
    # output shape
    assert logits.shape == (2, 3, config.vocab_size)
    
    # No NaN / inf
    assert torch.isfinite(logits).all()
    
    # parameter count
    total = sum(p.numel() for p in model.parameters())
    
    assert total == 9_934_608

def test_causal_attention():
    torch.manual_seed(42)
    
    config = GPTConfig()
    model = GPT(config)
    model.eval()
    
    ids_1 = torch.tensor([[10, 20, 30]])
    ids_2 = torch.tensor([[10, 20, 99]])
    
    with torch.no_grad():
        logits_1 = model(ids_1)
        logits_2 = model(ids_2)
    
    assert torch.allclose(logits_1[:, 1, :], logits_2[:, 1, :], atol=1e-6)
    