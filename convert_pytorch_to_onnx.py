from model import SimpleCNN
import torch

model = SimpleCNN()

checkpoint_path = 'simpleCNN.pt'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])


#convert to onnx
dummy_input = torch.randn(1, 3, 28, 28)
torch.onnx.export(model, dummy_input, 'simpleCNN.onnx', opset_version=11)
