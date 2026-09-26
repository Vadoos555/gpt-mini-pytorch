import torch


def generate(model, ids, max_new_tokens, temperature=1.0):
    model.eval()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(ids)
            
        logits = logits[:, -1, :]
        logits = logits / temperature
        
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        ids = torch.cat(
            [ids, next_token],
            dim=1
        )
    
    return ids


if __name__ == '__main__':

    logits = torch.tensor([
        [5.0, 4.0, 2.0, 0.5]
    ])

    for temperature in [0.5, 1.0, 2.0]:
        probs = torch.softmax(logits / temperature, dim=-1)

        print(temperature, probs)
        