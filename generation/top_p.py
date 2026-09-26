import torch


def generate(model, ids, max_new_tokens, top_p=0.9, temperature=1.0):
    model.eval()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(ids)
        
        logits = logits[:, -1, :]
        logits = logits / temperature
        
        probs = torch.softmax(logits, dim=-1)
        
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        
        sorted_probs[remove] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        
        probs = torch.zeros_like(probs)
        probs.scatter_(-1, sorted_indices, sorted_probs)
        
        next_token = torch.multinomial(probs, num_samples=1)
        
        ids = torch.cat(
            [ids, next_token],
            dim=1
        )
        
    return ids


if __name__ == "__main__":
    probs = torch.tensor([
        [0.40, 0.05, 0.02, 0.25, 0.01, 0.15, 0.03, 0.08, 0.005, 0.005]
    ])

    top_p = 0.90

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    remove = cumulative_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    sorted_probs[remove] = 0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    new_probs = torch.zeros_like(probs)
    new_probs.scatter_(-1, sorted_indices, sorted_probs)

    print("original:")
    print(probs)

    print("sorted:")
    print(sorted_probs)

    print("new:")
    print(new_probs)

    print("sum:")
    print(new_probs.sum())
