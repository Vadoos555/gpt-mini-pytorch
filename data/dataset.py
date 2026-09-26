import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class GPTDataset(Dataset):
    def __init__(self, tokens, context_size):
        self.tokens = tokens
        self.context_size = context_size
    
    def __len__(self):
        return len(self.tokens) - self.context_size
    
    def __getitem__(self, index):
        start = index
        end = start + self.context_size
        
        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        
        return (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))


if __name__ == '__main__':
    tokens = [10, 20, 30, 40, 50, 60]
    
    dataset = GPTDataset(tokens, context_size=3)
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    for x, y in loader:
        print("x:")
        print(x)

        print("y:")
        print(y)

        print("x shape:", x.shape)
        print("y shape:", y.shape)
        print('-------------------')
        