import torch
from torch import nn
import torch.nn.functional as F

# Convolutional Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)

# Dense Layer
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.conv = ConvLayer(in_channels, growth_rate)

    def forward(self, x):
        out = self.conv(x)
        return torch.cat([x, out], 1)

# Residual Dense Block (RDB)
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(ResidualDenseBlock, self).__init__()
        self.layers = nn.ModuleList([DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)])
        self.lff = ConvLayer(in_channels + num_layers * growth_rate, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.lff(x) + x

# Global Feature Fusion
class GlobalFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(GlobalFeatureFusion, self).__init__()
        self.conv1x1 = ConvLayer(in_channels * num_blocks, out_channels, kernel_size=1, padding=0)
        self.conv3x3 = ConvLayer(out_channels, out_channels)

    def forward(self, x):
        x = torch.cat(x, 1)
        return self.conv3x3(self.conv1x1(x))

# RDN Model
class RDN(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, growth_rate, num_layers):
        super(RDN, self).__init__()
        self.G0 = in_channels
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # Shallow feature extraction
        self.SFENet1 = ConvLayer(in_channels, self.G0)
        self.SFENet2 = ConvLayer(self.G0, self.G0)

        # Residual dense blocks
        self.RDBs = nn.ModuleList([ResidualDenseBlock(self.G0, self.G, self.C) for _ in range(self.D)])

        # Global feature fusion
        self.GFF = GlobalFeatureFusion(self.G0, self.G0, self.D)

        # Final convolution layer
        self.output_conv = ConvLayer(self.G0, out_channels)

    def forward(self, x):
        shallow_features = self.SFENet2(self.SFENet1(x))

        rdb_outs = [shallow_features]
        for rdb in self.RDBs:
            rdb_outs.append(rdb(rdb_outs[-1]))

        gff_out = self.GFF(rdb_outs)
        return self.output_conv(gff_out + shallow_features)

