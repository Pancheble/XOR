import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'epochs': 10000,
    'learning_rate': 1e-1
}