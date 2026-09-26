import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        example = self.data[index]
        
        user_text = (
            "<|user|>\n"
            + example["instruction"]
            + "\n"
            + "<|assistant|>\n"
        )
        
        response_text = example['response']
        
        user_ids = self.tokenizer.encode(user_text)
        response_ids = self.tokenizer.encode(response_text)
        
        input_ids = user_ids + response_ids
        target_ids = ([-100] * len(user_ids) + response_ids)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        
        return input_ids, target_ids
        