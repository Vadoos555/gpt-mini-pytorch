import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        
        return x


if __name__ == '__main__':
    torch.manual_seed(42)
    
    mlp = MLP(d_model=256)
    
    x = torch.randn(2, 3, 256)
    y = mlp(x)
    
    print('x: ', x.shape)
    print('y: ', y.shape)
    