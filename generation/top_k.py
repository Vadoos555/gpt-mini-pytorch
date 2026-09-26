import torch


def generate(model, ids, max_new_tokens, k=5, temperature=1.0):
    model.eval()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(ids)
        
        logits = logits[:, -1, :]
        logits = logits / temperature
        
        values, _ = torch.topk(logits, k=k, dim=-1)
        threshold = values[:, -1:]
        
        logits = logits.masked_fill(
            logits < threshold,
            float('-inf')
        )
        
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        ids = torch.cat(
            [ids, next_token],
            dim=1
        )
    
    return ids


if __name__ == "__main__":
    logits = torch.tensor([
        [1.2, 5.8, 2.1, 7.3, 0.4, 4.9, 1.7, 6.2, 3.5, 5.1]
    ])

    k = 5

    values, _ = torch.topk(
        logits,
        k=k,
        dim=-1
    )

    threshold = values[:, -1:]

    print("top-k values:")
    print(values)

    print("threshold:")
    print(threshold)

    logits = logits.masked_fill(
        logits < threshold,
        float("-inf")
    )

    print("masked logits:")
    print(logits)

    probs = torch.softmax(logits, dim=-1)

    print("probs:")
    print(probs)
    