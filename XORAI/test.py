import torch
from model import Model
from config import device

def test(model_path, input_data):
    model = Model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        outputs = model(input_data)
    
    return outputs

if __name__ == '__main__':
    model_path = r'XORAI/xormodel.pth'
    input_data = torch.tensor(list(map(float, input().split())))
    
    print(test(model_path=model_path, input_data=input_data))