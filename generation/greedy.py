import torch


def generate(model, ids, max_new_tokens):
    model.eval()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(ids)
            
        next_token = logits[:, -1, :].argmax(dim=-1)
        
        ids = torch.cat(
            [ids, next_token.unsqueeze(1)],
            dim=1
        )
        
    return ids
