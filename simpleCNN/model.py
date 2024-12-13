import torch 
import torch.nn as nn
import torch.nn.functional as F
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        #define layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) 
        self.conv3 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3)

        #define maxpooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #define upsampling layer
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = x
        x2 = x
        x3 = x
        x4 = x

        #branch 1
        x1 = F.relu(self.conv1(x1))

        #branch 2
        x2 = F.relu(self.conv1(x2))
        x2 = F.relu(self.conv2(x2))

        #branch 3
        x3 = F.relu(self.conv1(x3))
        x3 = F.relu(self.conv3(x3))

        #branch 4
        x4 = F.relu(self.maxpool(x4))
        x4 = F.relu(self.conv1(x4))
        x4 = self.upsampling(x4)

        #concatenate
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        return x


model = SimpleCNN()

input = torch.randn(1, 3, 28, 28)
output = model(input)
print(output.shape)

#save the checkpoint and state dict
checkpoint_path = 'simpleCNN.pt'
torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)