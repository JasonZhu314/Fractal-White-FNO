import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.baselines.fno import SpectralConv1d, compl_mul1d, add_padding, remove_padding, _get_act


class UNet1d(nn.Module):
    """1D U-Net for local feature extraction in U-FNO layers.

    3-level encoder-decoder with skip connections (concatenation).
    Adapted from the 3D U_net in ufno.py.

    Input/output shape: (batch, channels, spatial)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0):
        super(UNet1d, self).__init__()
        pad = kernel_size // 2

        # Encoder
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size,
                               stride=2, padding=pad)
        self.bn1 = nn.BatchNorm1d(in_channels)

        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size,
                               stride=2, padding=pad)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.conv2_1 = nn.Conv1d(in_channels, in_channels, kernel_size,
                                 stride=1, padding=pad)
        self.bn2_1 = nn.BatchNorm1d(in_channels)

        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size,
                               stride=2, padding=pad)
        self.bn3 = nn.BatchNorm1d(in_channels)
        self.conv3_1 = nn.Conv1d(in_channels, in_channels, kernel_size,
                                 stride=1, padding=pad)
        self.bn3_1 = nn.BatchNorm1d(in_channels)

        # Decoder
        self.deconv2 = nn.ConvTranspose1d(in_channels, in_channels, kernel_size,
                                          stride=2, padding=pad, output_padding=1)
        self.bn_d2 = nn.BatchNorm1d(in_channels)

        self.deconv1 = nn.ConvTranspose1d(in_channels * 2, in_channels, kernel_size,
                                          stride=2, padding=pad, output_padding=1)
        self.bn_d1 = nn.BatchNorm1d(in_channels)

        self.deconv0 = nn.ConvTranspose1d(in_channels * 2, in_channels, kernel_size,
                                          stride=2, padding=pad, output_padding=1)
        self.bn_d0 = nn.BatchNorm1d(in_channels)

        # Output: concat with original input -> project to out_channels
        self.output_layer = nn.Conv1d(in_channels * 2, out_channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout_rate)
        self.act = F.leaky_relu

    def forward(self, x):
        # x: (batch, channels, spatial)

        # Encoder
        out_conv1 = self.act(self.bn1(self.conv1(x)))                          # /2
        out_conv2 = self.act(self.bn2(self.conv2(out_conv1)))
        out_conv2 = self.act(self.bn2_1(self.conv2_1(out_conv2)))              # /4
        out_conv3 = self.act(self.bn3(self.conv3(out_conv2)))
        out_conv3 = self.dropout(self.act(self.bn3_1(self.conv3_1(out_conv3)))) # /8

        # Decoder
        out_deconv2 = self.act(self.bn_d2(self.deconv2(out_conv3)))            # /4
        # Trim to match skip connection size
        out_deconv2 = out_deconv2[..., :out_conv2.shape[-1]]
        concat2 = torch.cat([out_conv2, out_deconv2], dim=1)

        out_deconv1 = self.act(self.bn_d1(self.deconv1(concat2)))              # /2
        out_deconv1 = out_deconv1[..., :out_conv1.shape[-1]]
        concat1 = torch.cat([out_conv1, out_deconv1], dim=1)

        out_deconv0 = self.act(self.bn_d0(self.deconv0(concat1)))              # /1
        out_deconv0 = out_deconv0[..., :x.shape[-1]]
        concat0 = torch.cat([x, out_deconv0], dim=1)

        out = self.output_layer(concat0)
        return out


class UFNO1d(nn.Module):
    """1D U-shaped Fourier Neural Operator.

    Combines spectral convolutions (global Fourier features) with U-Net branches
    (local spatial features) in a subset of layers, following the U-FNO paper.

    Architecture:
        fc0 (lift) -> [standard FNO layers] -> [U-FNO layers] -> fc1 -> fc2 (project)

    Standard FNO layer: SpectralConv1d + Conv1d bypass + activation
    U-FNO layer:        SpectralConv1d + Conv1d bypass + UNet1d + activation

    Input shape:  (batch, n_x, in_dim)
    Output shape: (batch, n_x, out_dim)
    """

    def __init__(self, modes, width=128, n_fno_layers=3, n_ufno_layers=3,
                 fc_dim=128, in_dim=2, out_dim=1, act='gelu', pad_ratio=0,
                 cnn_kernel_size=1, unet_kernel_size=3, dropout_rate=0):
        super(UFNO1d, self).__init__()

        self.width = width
        self.pad_ratio = pad_ratio
        self.fc_dim = fc_dim
        self.n_fno_layers = n_fno_layers
        self.n_ufno_layers = n_ufno_layers
        n_total = n_fno_layers + n_ufno_layers

        assert len(modes) == n_total, \
            f"modes list length ({len(modes)}) must equal n_fno_layers + n_ufno_layers ({n_total})"

        # Lift
        self.fc0 = nn.Linear(in_dim, width)

        # Spectral convolution layers (all layers)
        self.sp_convs = nn.ModuleList([
            SpectralConv1d(width, width, modes[i])
            for i in range(n_total)
        ])

        # 1x1 conv bypass layers (all layers)
        self.ws = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=cnn_kernel_size,
                      padding=cnn_kernel_size // 2)
            for _ in range(n_total)
        ])

        # U-Net branches (only for U-FNO layers)
        self.unets = nn.ModuleList([
            UNet1d(width, width, kernel_size=unet_kernel_size,
                   dropout_rate=dropout_rate)
            for _ in range(n_ufno_layers)
        ])

        # Project
        if fc_dim > 0:
            self.fc1 = nn.Linear(width, fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(width, out_dim)

        self.act = _get_act(act)

    def forward(self, x):
        """
        Input:  (batch, n_x, in_dim)
        Output: (batch, n_x, out_dim)
        """
        n_total = self.n_fno_layers + self.n_ufno_layers

        x = self.fc0(x)
        x = x.permute(0, 2, 1)  # (batch, width, n_x)
        pad_nums = [math.floor(self.pad_ratio * x.shape[-1])]
        x = add_padding(x, pad_nums=pad_nums)

        for i in range(n_total):
            x1 = self.sp_convs[i](x)
            x2 = self.ws[i](x)
            if i >= self.n_fno_layers:
                # U-FNO layer: add U-Net branch
                x3 = self.unets[i - self.n_fno_layers](x)
                x = x1 + x2 + x3
            else:
                # Standard FNO layer
                x = x1 + x2
            if self.act is not None and i != n_total - 1:
                x = self.act(x)

        x = remove_padding(x, pad_nums=pad_nums)
        x = x.permute(0, 2, 1)  # (batch, n_x, width)

        fc_dim = self.fc_dim if hasattr(self, 'fc_dim') else 1
        if fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)
        return x

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
