import torch

from generation.greedy import generate


def evaluate(model, tokenizer, instruction, max_new_tokens=50):
    model.eval()
    
    prompt = (
        "<|user|>\n"
        + instruction
        + "\n"
        + "<|assistant|>\n"
    )
    
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    
    output_ids = generate(model=model, ids=input_ids, max_new_tokens=max_new_tokens)
      
    output_ids = output_ids[0].tolist()
    output_text = tokenizer.decode(output_ids)
    
    response = output_text.split("<|assistant|>", 1)[1]
    
    return response.strip()
