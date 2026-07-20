"""
v3 Backbone: DualPathV3 - MoffatPSF + MultiOrientStrip + SKFusion.
Self-contained, torch-only implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoffatPSFChannel(nn.Module):
    """Learnable multi-scale Moffat profile kernels with scale attention."""

    def __init__(self, ch, K=4, max_k=9):
        super().__init__()
        self.K = K
        self.max_k = max_k if max_k % 2 == 1 else max_k + 1
        self.alpha = nn.Parameter(torch.rand(K) * 2 + 1)
        self.beta = nn.Parameter(torch.rand(K) * 1.5 + 1.5)
        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, K, 1), nn.Softmax(dim=1))
        self.proj = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.max_k
        ys = torch.arange(k, device=x.device).float() - (k - 1) / 2
        xs = torch.arange(k, device=x.device).float() - (k - 1) / 2
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        r2 = gy ** 2 + gx ** 2

        kernels = []
        for i in range(self.K):
            a = F.softplus(self.alpha[i]) + 0.5
            b = F.softplus(self.beta[i]) + 1.0
            kern = (1 + r2 / (a * a)) ** (-b)
            kern = kern / kern.sum()
            kernels.append(kern)
        kernel = torch.stack(kernels)[None, :, None, :, :]

        attn = self.scale_attn(x)
        kernel = (kernel * attn[:, :, None, :, :]).sum(dim=1, keepdim=True)
        kernel = kernel.squeeze(2).expand(-1, C, -1, -1).reshape(B * C, 1, k, k)
        x_pad = F.pad(x, [k // 2] * 4, mode='replicate')
        out = F.conv2d(x_pad.reshape(1, B * C, H + k - 1, W + k - 1),
                       kernel, groups=B * C, padding=0)
        out = out.reshape(B, C, H, W)
        return self.proj(out) + x


class MultiOrientStripChannel(nn.Module):
    """Fixed 4-direction strip convolution with orientation attention."""

    def __init__(self, ch, strip_len=15, strip_w=5):
        super().__init__()
        self.angles = [0, 45, 90, 135]
        half = (strip_len - 1) // 2
        self.pad_h = (half, half, 0, 0)
        self.pad_w = (0, 0, half, half)
        self.strip_h = nn.ModuleList([
            nn.Conv2d(ch, ch, (strip_w, strip_len),
                      padding=(strip_w // 2, 0), groups=ch, bias=False)
            for _ in range(4)])
        self.strip_v = nn.ModuleList([
            nn.Conv2d(ch, ch, (strip_len, strip_w),
                      padding=(0, strip_w // 2), groups=ch, bias=False)
            for _ in range(4)])
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, 4, 1), nn.Softmax(dim=1))
        self.proj = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.ReLU(inplace=True))

    def _rotate(self, x, angle_deg):
        if angle_deg == 0:
            return x, lambda y: y
        rad = angle_deg * 3.1415926535 / 180.0
        cos_a, sin_a = torch.cos(torch.tensor(rad)), torch.sin(torch.tensor(rad))
        theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
                             device=x.device, dtype=torch.float32)
        theta = theta.unsqueeze(0).expand(x.shape[0], -1, -1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        rot = F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')
        theta_inv = torch.tensor([[cos_a, sin_a, 0], [-sin_a, cos_a, 0]],
                                 device=x.device, dtype=torch.float32)
        theta_inv = theta_inv.unsqueeze(0).expand(x.shape[0], -1, -1)

        def unrotate(y):
            return F.grid_sample(
                y, F.affine_grid(theta_inv, y.shape, align_corners=False),
                align_corners=False, padding_mode='zeros')

        return rot, unrotate

    def forward(self, x):
        outs = []
        for i, ang in enumerate(self.angles):
            rot, unrot = self._rotate(x, ang)
            h_out = self.strip_h[i](F.pad(rot, self.pad_h, mode='constant', value=0))
            v_out = self.strip_v[i](F.pad(rot, self.pad_w, mode='constant', value=0))
            outs.append(unrot(h_out + v_out))
        stacked = torch.stack(outs, dim=1)
        attn = self.attn(x).unsqueeze(2)
        fused = (stacked * attn).sum(dim=1)
        return self.proj(fused) + x


class SKFusion(nn.Module):
    """Channel + spatial selective fusion."""

    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(1, ch // reduction)
        self.ch_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch * 2, mid, 1),
            nn.ReLU(inplace=True), nn.Conv2d(mid, ch * 2, 1))
        self.sp_attn = nn.Sequential(
            nn.Conv2d(ch * 2, mid, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, 3, padding=1))
        self.proj = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.ReLU(inplace=True))

    def forward(self, psf_feat, strip_feat, identity):
        concat = torch.cat([psf_feat, strip_feat], dim=1)
        ch_w = self.ch_attn(concat).reshape(-1, 2, psf_feat.shape[1], 1, 1)
        ch_w = F.softmax(ch_w, dim=1)
        sp_w = F.softmax(self.sp_attn(concat), dim=1)
        fuse_ch = psf_feat * ch_w[:, 0] + strip_feat * ch_w[:, 1]
        fuse_sp = psf_feat * sp_w[:, 0:1] + strip_feat * sp_w[:, 1:2]
        return self.proj(fuse_ch + fuse_sp) + identity


class DualBlockV3(nn.Module):
    """MoffatPSF + MultiOrientStrip + SKFusion, followed by downsampling."""

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.psf = MoffatPSFChannel(in_ch)
        self.strip = MultiOrientStripChannel(in_ch)
        self.fuse = SKFusion(in_ch)
        self.identity = nn.Identity()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        identity = self.identity(x)
        psf_out = self.psf(x)
        strip_out = self.strip(x)
        fused = self.fuse(psf_out, strip_out, identity)
        return self.down(fused)


class SingleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride) if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip(x)


class Stem(nn.Module):
    def __init__(self, in_ch=1, out_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out + self.skip(x)


class DualPathBackbone(nn.Module):
    """Stem -> 2 DualBlockV3 -> 2 SingleBlock -> 4 feature scales."""

    def __init__(self, in_chans=1):
        super().__init__()
        self.stem = Stem(in_chans, 32)
        self.stage1 = DualBlockV3(32, 64)
        self.stage2 = DualBlockV3(64, 128)
        self.stage3 = SingleBlock(128, 256)
        self.stage4 = SingleBlock(256, 512)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.stage1(c1)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return [c2, c3, c4, c5], None, None
