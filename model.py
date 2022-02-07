import torch

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                torch.nn.ReLU()
                )
        
    def forward(self, x):
        return self.conv(x)
    
class UNet(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]): # [64, 128, 256, 512]
        super().__init__()
        
        self.layer1 = DoubleConv(1, features[0])
        self.layer2 = DoubleConv(features[0], features[1])
        self.layer3 = DoubleConv(features[1], features[2])
        self.layer4 = DoubleConv(features[2], features[3])
        
        self.layer5 = DoubleConv(features[3] + features[2], features[2])
        self.layer6 = DoubleConv(features[2] + features[1], features[1])
        self.layer7 = DoubleConv(features[1] + features[0], features[0])
        self.layer8 = torch.nn.Conv2d(features[0], out_channels, 1)
        
        self.max_pool = torch.nn.MaxPool2d(2)
        
    def forward(self, x):
        x1 = self.layer1(x)
        x1_pool = self.max_pool(x1)
        
        x2 = self.layer2(x1_pool)
        x2_pool = self.max_pool(x2)
        
        x3 = self.layer3(x2_pool)
        x3_pool = self.max_pool(x3)
        
        x4 = self.layer4(x3_pool)
        
        x5 = torch.nn.Upsample(scale_factor=2, mode='bilinear')(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.layer5(x5)
        
        x6 = torch.nn.Upsample(scale_factor=2, mode='bilinear')(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.layer6(x6)
        
        x7 = torch.nn.Upsample(scale_factor=2, mode='bilinear')(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.layer7(x7)
        
        output = self.layer8(x7)
        return output

"""        
model = UNet()
random_input = torch.randn(1, 1, 256, 256)
output = model(random_input)
assert output.shape == torch.Size([1, 1, 256, 256])
"""