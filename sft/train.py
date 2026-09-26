import torch
import torch.nn as nn


def train_sft(model, dataloader, optimizer):
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.train()
    total_loss = 0.0
        
    for input_ids, target_ids in dataloader:
        optimizer.zero_grad()
            
        # shift
        x = input_ids[:, :-1]
        y = target_ids[:, 1:]
            
        # forward
        logits = model(x)
            
        logits = logits.reshape(-1, logits.size(-1))
        targets = y.reshape(-1)
            
        loss = loss_fn(logits, targets)
            
        # backpropagation
        loss.backward()
            
        # update weights
        optimizer.step()
            
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
        
    return avg_loss
