import torch
import torch.nn as nn
import math
from compressai.layers import GDN
from torch import Tensor

class ACBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACBlock, self).__init__()
        self.conv1x9 = nn.Sequential(nn.Conv2d(in_channels, out_channels, (1, 9), 1, (0, 4)))
        self.conv9x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, (9, 1), 1, (4, 0)))
        self.conv3x3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 3), 1, (1, 1)))
        self.fuse = nn.Conv2d(3*in_channels, out_channels, 1, 1, 0)
        self.attn = eca_layer(3*in_channels)
        
    def forward(self, x):
        conv9x1 = self.conv9x1(x)
        conv1x9 = self.conv1x9(x)
        conv3x3 = self.conv3x3(x)
        buffer = torch.cat((conv1x9,conv9x1,conv3x3),dim=1)
        buffer = self.attn(buffer)
        buffer_fuse = self.fuse(buffer)
        return x + buffer_fuse
    

class AClayer_stride(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.acblock = ACBlock(in_ch, out_ch)
        self.gdn = GDN(out_ch)
        self.gelu = nn.GELU()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride)
        else:
            self.skip = None
        
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv(x)
        out = self.acblock(out)
        out = self.gelu(out)
        out = self.acblock(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out

def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )

def deconv(in_channels, out_channels, kernel_size=3, stride=2):     # SN -1 + k - 2p
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


class AClayer_upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)
        self.acblock = ACBlock(in_ch, out_ch)
        self.gdn = GDN(out_ch)
        self.gelu = nn.GELU()
       
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.acblock(out)
        out = self.gelu(out)
        out = self.acblock(out)
        out = self.gdn(out)
        identity = self.upsample(x)

        out += identity
        return out


class AClayer_deconv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = deconv(in_ch, out_ch, 3, upsample)
        self.upsample = deconv(in_ch, out_ch, 3, upsample)
        self.acblock = ACBlock(in_ch, out_ch)
        self.gdn = GDN(out_ch)
        self.gelu = nn.GELU()
       
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.acblock(out)
        out = self.gelu(out)
        out = self.acblock(out)
        out = self.gdn(out)
        identity = self.upsample(x)

        out += identity
        return out



class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, c, b=1, gamma=2):
        super(eca_layer, self).__init__()
        
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding= int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)