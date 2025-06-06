import torch
from config import device

x = torch.tensor(
    [[0., 0.],
     [0., 1.],
     [1., 0.],
     [1., 1.]]
).to(device=device)

y = torch.tensor(
    [[0.],
     [1.],
     [1.],
     [0.]]
).to(device=device)