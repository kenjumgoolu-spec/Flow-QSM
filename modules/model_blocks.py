import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

class BackboneModulation(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_conv = nn.Conv3d(1, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_mean = x.mean(dim=1, keepdim=True)
        alpha = self.sigmoid(self.spatial_conv(x_mean))
        C = x.shape[1]
        x_mod = x.clone()
        x_mod[:, :C//2] = x[:, :C//2] * alpha
        return x_mod

class SpectralModulation(nn.Module):
    def __init__(self, base_size=(32, 32, 32)):
        super().__init__()
        self.mask_param = nn.Parameter(torch.zeros(1, 1, *base_size))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, D, H, W = x.shape
        orig_dtype = x.dtype
        x_freq = fft.fftn(x.float(), dim=(-3, -2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
        M_l = F.interpolate(self.mask_param, size=(D, H, W), mode='trilinear', align_corners=False)
        M_l = self.sigmoid(M_l)
        if orig_dtype != torch.float32:
            M_l = M_l.to(torch.float32)
        x_freq = x_freq * M_l
        x_freq = fft.ifftshift(x_freq, dim=(-3, -2, -1))
        x_hat = fft.ifftn(x_freq, dim=(-3, -2, -1)).real
        if orig_dtype != torch.float32:
            x_hat = x_hat.to(orig_dtype)
        return x_hat