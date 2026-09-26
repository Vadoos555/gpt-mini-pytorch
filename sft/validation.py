import torch
import torch.nn as nn


def validate_sft(model, dataloader):
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.eval()
    total_los = 0.0
    
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            # shift
            x = input_ids[:, :-1]
            y = target_ids[:, 1:]
            
            # forward
            logits = model(x)
            
            logits = logits.reshape(-1, logits.size(-1))
            targets = y.reshape(-1)
            
            loss = loss_fn(logits, targets)
            
            total_los += loss.item()
    
    avg_loss = total_los / len(dataloader)
    
    return avg_loss
