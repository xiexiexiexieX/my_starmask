"""
MAINet v4 backbone.

This version uses three conservative morphology priors:
  1. Gated residual PSF enhancement
  2. Near-collinear deformable orientation strip sampling
  3. Image-level morphology-guided residual fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualContextDownsample(nn.Module):
    """Cheap residual downsampling that grows the morphology receptive field."""

    def __init__(self, ch):
        super().__init__()
        gn = max(1, min(8, ch))
        self.main = nn.Sequential(
            nn.Conv2d(ch, ch, 5, stride=2, padding=2, groups=ch),
            nn.GroupNorm(gn, ch),
            nn.ReLU(inplace=True),
        )
        self.skip = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        return F.relu(self.main(x) + self.skip(x), inplace=True)


class ResidualContextRefine(nn.Module):
    """Dilated residual refinement without discarding tiny-source detail."""

    def __init__(self, ch):
        super().__init__()
        gn = max(1, min(8, ch))
        self.main = nn.Sequential(
            nn.Conv2d(ch, ch, 5, padding=4, dilation=2, groups=ch),
            nn.GroupNorm(gn, ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return F.relu(self.main(x) + x, inplace=True)


class GlobalMorphologyContext(nn.Module):
    """Predict an image-level point/streak gate from geometry-aware context."""

    def __init__(self, ch, hidden=32):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualContextDownsample(ch),
            ResidualContextDownsample(ch),
            ResidualContextRefine(ch),
        )
        self.register_buffer(
            'sobel_x',
            torch.tensor([[-1.0, 0.0, 1.0],
                          [-2.0, 0.0, 2.0],
                          [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3),
            persistent=False)
        self.register_buffer(
            'sobel_y',
            torch.tensor([[-1.0, -2.0, -1.0],
                          [0.0, 0.0, 0.0],
                          [1.0, 2.0, 1.0]]).view(1, 1, 3, 3),
            persistent=False)
        self.shared = nn.Sequential(
            nn.Linear(ch * 3 + 3, hidden),
            nn.ReLU(inplace=True),
        )
        self.gate_head = nn.Linear(hidden, 1)
        self.direction_head = nn.Linear(hidden, 2)
        nn.init.normal_(self.gate_head.weight, std=0.01)
        nn.init.constant_(self.gate_head.bias, 0.0)
        nn.init.normal_(self.direction_head.weight, std=0.01)
        with torch.no_grad():
            self.direction_head.bias.copy_(torch.tensor([1.0, 0.0]))

    def _structure_descriptor(self, x):
        signal = x.mean(dim=1, keepdim=True)
        sobel_x = self.sobel_x.to(dtype=signal.dtype)
        sobel_y = self.sobel_y.to(dtype=signal.dtype)
        gx = F.conv2d(signal, sobel_x, padding=1)
        gy = F.conv2d(signal, sobel_y, padding=1)
        jxx = gx.square().mean(dim=(1, 2, 3))
        jyy = gy.square().mean(dim=(1, 2, 3))
        jxy = (gx * gy).mean(dim=(1, 2, 3))
        energy = (jxx + jyy).clamp_min(1e-4)
        cos2 = (jxx - jyy) / energy
        sin2 = (2.0 * jxy) / energy
        coherence = torch.sqrt(cos2.square() + sin2.square() + 1e-6)
        return torch.stack([cos2, sin2, coherence], dim=1)

    def forward(self, x):
        encoded = self.encoder(x)
        avg = F.adaptive_avg_pool2d(encoded, 1).flatten(1)
        peak = F.adaptive_max_pool2d(encoded, 1).flatten(1)
        regional_peaks = F.adaptive_max_pool2d(encoded, 4)
        sparse = regional_peaks.flatten(2).mean(dim=2)
        structure = self._structure_descriptor(encoded)
        context = self.shared(
            torch.cat([avg, peak, sparse, structure], dim=1))
        gate_logit = self.gate_head(context).squeeze(1)
        direction = F.normalize(self.direction_head(context), dim=1, eps=1e-6)
        return gate_logit, direction


class PSFGatedEnhancement(nn.Module):
    """PSF prior as a gated residual enhancement, not a feature replacement."""

    def __init__(self, ch, K=4, max_k=9):
        super().__init__()
        self.K = K
        self.max_k = max_k if max_k % 2 == 1 else max_k + 1
        self.alpha = nn.Parameter(torch.rand(K) * 2 + 1)
        self.beta = nn.Parameter(torch.rand(K) * 1.5 + 1.5)
        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, K, 1),
            nn.Softmax(dim=1),
        )
        gn = max(1, min(8, ch))
        self.enhance = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.GroupNorm(gn, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.max_k
        ys = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
        xs = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        r2 = gy.square() + gx.square()

        kernels = []
        for i in range(self.K):
            a = F.softplus(self.alpha[i]).to(dtype=x.dtype) + 0.5
            b = F.softplus(self.beta[i]).to(dtype=x.dtype) + 1.0
            kern = (1 + r2 / (a * a)) ** (-b)
            kernels.append(kern / (kern.sum() + 1e-6))

        kernel = torch.stack(kernels)[None, :, None, :, :]
        attn = self.scale_attn(x)
        kernel = (kernel * attn[:, :, None]).sum(dim=1, keepdim=True)
        kernel = kernel.squeeze(2).expand(-1, C, -1, -1).reshape(B * C, 1, k, k)

        x_pad = F.pad(x, [k // 2] * 4, mode='replicate')
        smooth = F.conv2d(
            x_pad.reshape(1, B * C, H + k - 1, W + k - 1),
            kernel,
            groups=B * C,
        ).reshape(B, C, H, W)

        delta = self.enhance(smooth - x)
        gate = self.gate(torch.cat([x, smooth], dim=1))
        return x + gate * delta


class DeformableOrientationStrip(nn.Module):
    """
    Learned spatial offsets + attention aggregation.

    This replaces fixed multi-orientation rotate sampling. The branch samples a
    continuous local field and lets attention select useful positions along the
    implicit streak direction.
    """

    def __init__(self, ch, num_points=9, max_offset=6.0,
                 max_perp_offset=1.5, attention_floor=0.25):
        super().__init__()
        self.num_points = num_points
        self.max_offset = float(max_offset)
        self.max_perp_offset = float(max_perp_offset)
        self.attention_floor = float(attention_floor)
        gn = max(1, min(8, ch))
        self.residual_offset = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=gn),
            nn.GroupNorm(gn, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, num_points * 2, 3, padding=1),
        )
        self.local_direction = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=gn),
            nn.GroupNorm(gn, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 2, 1),
        )
        self.attn = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, num_points, 1),
        )
        self.value = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.GroupNorm(gn, ch),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid(),
        )

    def _base_grid(self, B, H, W, device, dtype):
        # Pixel-center grid for grid_sample(..., align_corners=False).
        ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) * (2.0 / H) - 1.0
        xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) * (2.0 / W) - 1.0
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([gx, gy], dim=-1)
        return grid.unsqueeze(0).expand(B, -1, -1, -1)

    def forward(self, x, global_direction=None):
        B, C, H, W = x.shape
        residual = torch.tanh(self.residual_offset(x)).view(
            B, self.num_points, 2, H, W)
        learned_weights = F.softmax(self.attn(x), dim=1)
        weights = ((1.0 - self.attention_floor) * learned_weights +
                   self.attention_floor / self.num_points)

        if global_direction is None:
            direction = F.normalize(
                self.local_direction(x), dim=1, eps=1e-6)
        else:
            direction = F.normalize(
                global_direction, dim=1, eps=1e-6)[:, :, None, None]
            direction = direction.expand(-1, -1, H, W)

        ux, uy = direction[:, 0], direction[:, 1]
        vx, vy = -uy, ux
        positions = torch.linspace(
            -self.max_offset, self.max_offset, self.num_points,
            device=x.device, dtype=x.dtype)[None, :, None, None]
        spacing = self.max_offset / max(self.num_points - 1, 1)
        parallel = positions + residual[:, :, 0] * spacing
        perpendicular = residual[:, :, 1] * self.max_perp_offset
        offset_x = parallel * ux[:, None] + perpendicular * vx[:, None]
        offset_y = parallel * uy[:, None] + perpendicular * vy[:, None]
        offsets = torch.stack([offset_x, offset_y], dim=-1)

        norm = x.new_tensor([2.0 / max(W, 1), 2.0 / max(H, 1)])
        offsets = offsets * norm
        base = self._base_grid(B, H, W, x.device, x.dtype)
        val = self.value(x)

        sampled = []
        for i in range(self.num_points):
            grid = base + offsets[:, i]
            s = F.grid_sample(
                val,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False,
            )
            sampled.append(s * weights[:, i:i + 1])

        agg = torch.stack(sampled, dim=0).sum(dim=0)
        delta = self.proj(agg)
        gate = self.gate(torch.cat([x, delta], dim=1))
        return x + gate * delta


# Backward-compatible alias for old ablation imports. The implementation is no
# longer fixed-orientation or rotate-based.
MultiOrientStripChannel = DeformableOrientationStrip
MoffatPSFChannel = PSFGatedEnhancement


class GatedResidualFusion(nn.Module):
    """Additive gated fusion. PSF and strip branches cooperate instead of compete."""

    def __init__(self, ch):
        super().__init__()
        gn = max(1, min(8, ch))
        self.weight = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 2, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.GroupNorm(gn, ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, psf_feat, strip_feat, identity, mode_gate=None):
        psf_delta = psf_feat - identity
        strip_delta = strip_feat - identity
        w = self.weight(torch.cat([identity, psf_delta, strip_delta], dim=1))
        if mode_gate is None:
            psf_prior = strip_prior = 1.0
        else:
            gate = mode_gate[:, :, None, None]
            # Keep a small contribution from the non-primary branch so an
            # imperfect gate cannot erase useful residual evidence.
            psf_prior = 0.05 + 0.95 * (1.0 - gate)
            strip_prior = 0.05 + 0.95 * gate
        fused = (identity + psf_prior * w[:, 0:1] * psf_delta +
                 strip_prior * w[:, 1:2] * strip_delta)
        return self.proj(fused) + identity


# Backward-compatible alias for old model loading code.
SKFusion = GatedResidualFusion


class DualBlockV3(nn.Module):
    """PSF residual gate + deformable strip + gated residual fusion."""

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        gn = max(1, min(8, out_ch))
        self.psf = PSFGatedEnhancement(in_ch)
        self.strip = DeformableOrientationStrip(in_ch)
        self.fuse = GatedResidualFusion(in_ch)
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(gn, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, mode_gate=None, global_direction=None):
        psf_out = self.psf(x)
        strip_out = self.strip(x, global_direction=global_direction)
        fused = self.fuse(psf_out, strip_out, x, mode_gate=mode_gate)
        return self.down(fused)


class SingleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        gn = max(1, min(8, out_ch))
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(gn, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(gn, out_ch),
            nn.ReLU(inplace=True),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride) if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip(x)


class Stem(nn.Module):
    def __init__(self, in_ch=1, out_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(max(1, min(8, out_ch)), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.norm(self.conv2(out)), inplace=True)
        return out + self.skip(x)


class DualPathBackbone(nn.Module):
    """Stem -> 2 DualBlockV3 -> 2 SingleBlock. Returns stride 2/4/8/16 features."""

    def __init__(self, in_chans=1):
        super().__init__()
        self.stem = Stem(in_chans, 32)
        self.context = GlobalMorphologyContext(32)
        self.stage1 = DualBlockV3(32, 64)
        self.stage2 = DualBlockV3(64, 128)
        self.stage3 = SingleBlock(128, 256)
        self.stage4 = SingleBlock(256, 512)

    def forward(self, x):
        c1 = self.stem(x)
        gate_logit, direction = self.context(c1)
        mode_gate = torch.sigmoid(gate_logit).unsqueeze(1)
        # Gate classification is explicitly supervised. Do not let the much
        # larger detector losses collapse it toward whichever branch is easier.
        routing_gate = mode_gate.detach() if self.training else mode_gate
        c2 = self.stage1(c1, mode_gate=routing_gate,
                         global_direction=direction)
        c3 = self.stage2(c2, mode_gate=routing_gate,
                         global_direction=direction)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        aux = {'gate_logit': gate_logit, 'direction': direction}
        return [c2, c3, c4, c5], aux, None
