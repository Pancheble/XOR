from model import Model
from config import device
from config import config
from data import x, y
import torch
import torch.nn as nn
import torch.optim as optim


model = Model().to(device=device)
criterion = nn.BCELoss()
optimizer = optim.Adamax(model.parameters(), lr=config['learning_rate'])

for epoch in range(config['epochs']):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    print(f'Epoch: {epoch}, Loss: {loss.item()}')
    
torch.save(model.state_dict(), 'xormodel.pth')