import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class Generator(BaseNetwork):
    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.layer1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.lc1 = LC(64)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=1)
        self.a1 = nn.MaxPool2d(2, 2)
        self.a2 = Residual(64, 64)
        self.layer2 = Residual(64, 64)
        self.lc2 = LC(64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.b1 = Residual(64, 128, same_shape=False)
        self.layer3 = Residual(128, 128)
        self.lc3 = LC(128)
        self.conv3 = nn.Conv2d(64, 128,  kernel_size=1)
        self.c1 = Residual(128, 256, same_shape=False)
        self.layer4 = Residual(256, 256)
        self.lc4 = LC(256)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1)
        self.d1 = Residual(256, 512, same_shape=False)
        self.layer5 = Residual(512, 512)
        self.layer6 = nn.Conv2d(512, 2, kernel_size=1)
        self.delayer1 = nn.ConvTranspose2d(2, 256, kernel_size=4, padding=1, stride=2)
        self.relu1 = nn.ELU(alpha=1.0)
        self.relu11 = nn.ELU(alpha=1.0)
        self.delayer2 = nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2)
        self.relu2 = nn.ELU(alpha=1.0)
        self.relu22 = nn.ELU(alpha=1.0)
        self.delayer3 = nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2)
        self.convs_1 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu3 = nn.ELU(alpha=1.0)
        self.relu33 = nn.ELU(alpha=1.0)
        self.delayer4 = nn.ConvTranspose2d(64, 64, kernel_size=4, padding=1, stride=2)
        self.convs_2 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu4 = nn.ELU(alpha=1.0)
        self.relu44 = nn.ELU(alpha=1.0)
        self.delayer5 = nn.ConvTranspose2d(64, 3, kernel_size=4, padding=1, stride=2)
        self.relu5 = nn.ELU(alpha=1.0)

    def forward(self, x):
        c1 = self.layer1(x)
        lc1 = self.lc1(c1)
        a1 = self.a1(c1)
        a2 = self.a2(a1)
        c2 = self.layer2(a2)
        lc2 = self.lc2(c2)
        b1 = self.b1(c2)
        c3 = self.layer3(b1)
        lc3 = self.lc3(c3)
        C1 = self.c1(c3)
        c4 = self.layer4(C1)
        lc4 = self.lc4(c4)
        d1 = self.d1(c4)
        c5 = self.layer5(d1)
        p51 = self.layer6(c5)
        p5 = self.relu11(self.conv4(lc4) + self.relu1(self.delayer1(p51)))
        p6 = self.relu22(self.conv3(lc3) + self.relu2(self.delayer2(p5)))
        p7 = self.relu33(self.conv2(lc2) + self.relu3(self.delayer3(p6)))
        p7_o = self.convs_1(p7)
        p8 = self.relu44(self.conv1(lc1) + self.relu4(self.delayer4(p7)))
        p8_o = self.convs_2(p8)
        p9 = self.relu5(self.delayer5(p8))
        return p5, p6, p7_o, p8_o, p9


class LC(nn.Module):
    def __init__(self, outer_channels):
        super(LC, self).__init__()
        channels = int(np.ceil(outer_channels / 2))
        self.conv = nn.Sequential(nn.Conv2d(outer_channels, channels, kernel_size=1),
                                  nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1),
                                  nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1))

    def forward(self, x):
        out = self.conv(x)
        return out


class Residual(nn.Module):
    def __init__(self, i_channels, o_channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2d(i_channels, o_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(o_channels, o_channels, kernel_size=3, padding=1)
        if not same_shape:
            self.conv3 = nn.Conv2d(i_channels, o_channels, kernel_size=1, stride=strides)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)

class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2,
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        new_channel = 1
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64 + new_channel, out_channels=128, kernel_size=4,
                                    stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128 + new_channel, out_channels=256, kernel_size=4,
                                    stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256 + new_channel, out_channels=512, kernel_size=4,
                                    stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4,
                                    stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        mask = x[:, -1:, ::]
        conv1 = self.conv1(x)
        mask = F.interpolate(mask, size=(mask.size(2) // 2, mask.size(3) // 2), mode='nearest')
        conv1 = torch.cat([conv1, mask], dim=1)
        conv2 = self.conv2(conv1)
        mask = F.interpolate(mask, size=(mask.size(2) // 2, mask.size(3) // 2), mode='nearest')
        conv2 = torch.cat([conv2, mask], dim=1)
        conv3 = self.conv3(conv2)
        mask = F.interpolate(mask, size=(mask.size(2) // 2, mask.size(3) // 2), mode='nearest')
        conv3 = torch.cat([conv3, mask], dim=1)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
