from torch import nn
import torch

class MSCAM(nn.Module):

    def __init__(self, channels=64, r=4):
        super(MSCAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = x * wei
        return xo

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda:0")

    x, residual= torch.ones(8,64, 32, 32).to(device),torch.ones(8,64, 32, 32).to(device)
    channels=x.shape[1]

    model=MSCAM(channels=channels)
    model=model.to(device).train()
    output = model(x)
    print(output.shape)