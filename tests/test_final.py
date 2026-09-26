import os

import torch

from config.model_config import GPTConfig
from model.gpt import GPT

from tokenizer.bpe import BPE, pre_tokenize

from sft.dataset import data
from sft.formatting import format_example
from sft.tokenizer_setup import train_tokenizer
from sft.evaluation import evaluate

from generation.greedy import generate


CHECKPOINT_PATH = "checkpoints/sft_best.pt"

# --------------------------------------------------
# 1. Tokenizer
# --------------------------------------------------

def test_tokenizer():
    tokenizer = train_tokenizer(data)
    text = 'What is Python?'
    
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    
    assert decoded == text
    
    print('PASS №1: tokenizer')
    

# --------------------------------------------------
# 2. GPT forward
# --------------------------------------------------

def test_model_forward():
    tokenizer = train_tokenizer(data)
    config = GPTConfig(vocab_size=len(tokenizer.vocab))
    model = GPT(config)
    
    ids = torch.tensor([[1, 2, 3, 4]])
    logits = model(ids)
    
    assert logits.shape == (1, 4, len(tokenizer.vocab))
    
    print('PASS №2: model forward')


# --------------------------------------------------
# 3. Generation
# --------------------------------------------------

def test_generation():
    tokenizer = train_tokenizer(data)
    config = GPTConfig(vocab_size=len(tokenizer.vocab))
    model = GPT(config)
        
    ids = torch.tensor([[1, 2, 3]])
    
    output = generate(model, ids, max_new_tokens=5)
    
    assert output.shape == (1, 8)
    
    print('PASS №3: generation')


# --------------------------------------------------
# 4. Checkpoint
# --------------------------------------------------

def test_checkpoint():
    assert os.path.exists(CHECKPOINT_PATH)
    
    tokenizer = train_tokenizer(data)
    config = GPTConfig(vocab_size=len(tokenizer.vocab))
    model = GPT(config)
    
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state'])
    
    assert checkpoint['epoch'] > 0
    assert checkpoint['val_loss'] > 0
    
    print('PASS №4: checkpoint')


# --------------------------------------------------
# 5. Evaluation
# --------------------------------------------------

def test_evaluation():
    tokenizer = train_tokenizer(data)
    config = GPTConfig(vocab_size=len(tokenizer.vocab))
    model = GPT(config)
        
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state'])
    
    response = evaluate(model, tokenizer, 'What is Python?', max_new_tokens=10)
    
    assert isinstance(response, str)
    assert len(response) > 0
    
    print('PASS №5: evaluation')

# --------------------------------------------------
# Run all tests
# --------------------------------------------------

if __name__ == '__main__':
    print()
    print('GPT Mini Final Test')
    print('='*30)
    
    test_tokenizer()
    test_model_forward()
    test_generation()
    test_checkpoint()
    test_evaluation()
    
    print()
    print('ALL FINAL TESTS PASSED')
    