
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr

class EDSR(nn.Module):
    def __init__(self, res_blocks, feature_ch, upscale_factor, res_scaling=0.1, bn=False, activation='relu'):
        super(EDSR, self).__init__()

        self.head_conv = nn.Conv2d(3, feature_ch, kernel_size=3, stride=1, padding=1)

        layers = [
            *[ResBlock(feature_ch, res_scaling, bn, activation) for i in range(res_blocks)],
            nn.Conv2d(feature_ch, feature_ch, kernel_size=3, stride=1, padding=1)
        ]

        self.sequences = nn.Sequential(*layers)

        self.upsample = Upsample(feature_ch, upscale_factor)
        self.tail_conv = nn.Conv2d(feature_ch, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.head_conv(x)
        res = self.sequences(y)
        out = self.upsample(res+y)
        out = self.tail_conv(out)

        return out

class Upsample(nn.Module):
    def __init__(self, feature_ch, upscale_factor):
        super(Upsample, self).__init__()

        assert upscale_factor in [2,3,4]
        self.upscale_factor = upscale_factor

        if upscale_factor == 2:
            self.conv1 = nn.Conv2d(feature_ch, feature_ch*4, kernel_size=3, stride=1, padding=1)
            self.shuffle = nn.PixelShuffle(2)
        elif upscale_factor == 3:
            self.conv1 = nn.Conv2d(feature_ch, feature_ch*9, kernel_size=3, stride=1, padding=1)
            self.shuffle = nn.PixelShuffle(3)
        elif upscale_factor == 4:
            self.conv1 = nn.Conv2d(feature_ch, feature_ch*4, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(feature_ch, feature_ch*4, kernel_size=3, stride=1, padding=1)
            self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.shuffle(y)
        if self.upscale_factor == 4:
            y = self.conv2(y)
            y = self.shuffle(y)
        return y

class ResBlock(nn.Module):
    def __init__(self, feature_ch, res_scaling, bn, activation):
        super(ResBlock, self).__init__()

        self.res_scaling = res_scaling
        self.bn = bn

        assert activation in ['relu']

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(feature_ch, feature_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(feature_ch, feature_ch, kernel_size=3, stride=1, padding=1)

        if bn:
            self.bn1 = nn.BatchNorm2d(feature_ch)
            self.bn2 = nn.BatchNorm2d(feature_ch)
         
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y) if self.bn else y
        y = self.act(y)
        y - self.conv2(y)
        y = self.bn2(y) if self.bn else y

        return x + (y * self.res_scaling)

def geometric_self_ensemble(net, data):
    flip = [[2,3], [3], [2]]

    output = net(data)
    for axis in flip:
        output += net(data.flip(axis)).flip(axis)

    return output / (len(flip)+1)

if __name__ == '__main__':
    x = torch.Tensor(16,3,12,12)
    net = EDSR(5, 64, 4)
    y = net(x)
    print(y.shape)
