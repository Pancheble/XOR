import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.output = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.output(x)
        return x