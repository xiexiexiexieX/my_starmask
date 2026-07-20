"""
v2 Backbone: DualPathV2 — 顶层 GlobalParamEstimator + PSF + Strip + GatedFusion + φ/gate 监督
=============================================================================================
完全自包含。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Stem(nn.Module):
    def __init__(self, in_ch=1, out_ch=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.fuse = nn.Conv2d(out_ch + in_ch, out_ch, 1)

    def forward(self, x):
        feat = self.conv(x); feat = torch.cat([feat, x], dim=1)
        return self.fuse(feat)


class GlobalParamEstimator(nn.Module):
    """收窄版：只估 φ(方向)+L(长度)+gate(点/线门控)。σ/θ 固定。"""
    def __init__(self, in_ch, hidden=64, attn_size=32, num_heads=4):
        super().__init__()
        self.attn_size = attn_size; self.hidden = hidden
        self.reduce = nn.Conv2d(in_ch, hidden, 1)
        self.attn   = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.norm   = nn.LayerNorm(hidden)
        self.g_norm = nn.BatchNorm1d(hidden)
        self.fc_gate = nn.Linear(hidden, 1)
        self.fc_phi  = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 2))
        self.fc_len  = nn.Linear(hidden, 1)

    def forward(self, x):
        B = x.shape[0]
        h = self.reduce(x); h = F.adaptive_avg_pool2d(h, self.attn_size)
        seq = h.flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(seq, seq, seq); seq = self.norm(seq + attn_out)
        g = self.g_norm(seq.mean(dim=1))
        gate_logit = self.fc_gate(g).squeeze(1)
        phi_sc = F.normalize(self.fc_phi(g), dim=1)
        length = F.softplus(self.fc_len(g)).squeeze(1) + 1.0
        phi = 0.5 * torch.atan2(phi_sc[:, 0], phi_sc[:, 1])
        sigma_fixed = torch.full((B,), 1.0, device=x.device, dtype=x.dtype)
        theta_zero  = torch.zeros(B, device=x.device, dtype=x.dtype)
        return {'sigma_x':sigma_fixed, 'sigma_y':sigma_fixed, 'theta':theta_zero,
                'phi':phi, 'length':length, 'phi_sc':phi_sc, 'gate_logit':gate_logit}


def scale_params(params, stride):
    if stride == 1: return params
    out = dict(params)
    out['sigma_x'] = params['sigma_x'] / stride
    out['sigma_y'] = params['sigma_y'] / stride
    out['length']  = params['length']  / stride
    return out


class PSFChannel(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.ch=channels; self.ks=kernel_size
        self.pointwise = nn.Conv2d(channels, channels, 1)

    def _build_kernels(self, sigma_x, sigma_y, theta):
        B=sigma_x.shape[0]; ks=self.ks; half=ks//2
        device,dtype=sigma_x.device,sigma_x.dtype
        y,x=torch.meshgrid(torch.arange(-half,half+1,device=device,dtype=dtype),
                           torch.arange(-half,half+1,device=device,dtype=dtype),indexing='ij')
        x=x.view(1,ks,ks); y=y.view(1,ks,ks)
        sx=sigma_x.view(B,1,1).clamp(0.5,half); sy=sigma_y.view(B,1,1).clamp(0.5,half)
        c=theta.cos().view(B,1,1); s=theta.sin().view(B,1,1)
        xr=x*c+y*s; yr=-x*s+y*c
        g=torch.exp(-0.5*((xr/sx)**2+(yr/sy)**2))
        return (g/(g.sum(dim=(-2,-1),keepdim=True)+1e-8)).unsqueeze(1)

    def forward(self, x, params):
        B,C,H,W=x.shape
        kernels=self._build_kernels(params['sigma_x'],params['sigma_y'],params['theta'])
        x_fold=x.reshape(1,B*C,H,W); k_fold=kernels.repeat(1,C,1,1).reshape(B*C,1,self.ks,self.ks)
        out=F.conv2d(x_fold,k_fold,padding=self.ks//2,groups=B*C).reshape(B,C,H,W)
        return x+self.pointwise(out)


class StripChannel(nn.Module):
    def __init__(self, channels, k_long_max=25, k_short_max=11, tau=2.0):
        super().__init__()
        self.ch=channels; self.KL=k_long_max if k_long_max%2==1 else k_long_max+1
        self.KS=k_short_max if k_short_max%2==1 else k_short_max+1; self.tau=tau
        self.conv_h=nn.Conv2d(channels,channels,(1,self.KL),padding=(0,self.KL//2),groups=channels,bias=False)
        self.conv_v=nn.Conv2d(channels,channels,(self.KS,1),padding=(self.KS//2,0),groups=channels,bias=False)
        self.pointwise=nn.Conv2d(channels,channels,1)
        self.register_buffer('pos_long',torch.arange(-(self.KL//2),self.KL//2+1).float())
        self.register_buffer('pos_short',torch.arange(-(self.KS//2),self.KS//2+1).float())

    def _soft_window(self, size_param, positions, full_extent):
        B=size_param.shape[0]; half=F.softplus(full_extent).view(B,1)
        return torch.sigmoid((half-positions.abs().view(1,-1))/self.tau)

    def _dynamic_depthwise(self, x, conv, win, is_horizontal):
        B,C,H,W=x.shape
        if is_horizontal:
            w=conv.weight.unsqueeze(0)*win.view(B,1,1,1,-1); K=self.KL
            w=w.reshape(B*C,1,1,K); pad=(0,K//2)
        else:
            w=conv.weight.unsqueeze(0)*win.view(B,1,1,-1,1); K=self.KS
            w=w.reshape(B*C,1,K,1); pad=(K//2,0)
        return F.conv2d(x.reshape(1,B*C,H,W),w,padding=pad,groups=B*C).reshape(B,C,H,W)

    def _rotate(self, x, angle):
        B,C,H,W=x.shape; cos,sin=angle.cos(),angle.sin()
        theta=torch.zeros(B,2,3,device=x.device,dtype=x.dtype)
        theta[:,0,0]=cos;theta[:,0,1]=-sin;theta[:,1,0]=sin;theta[:,1,1]=cos
        return F.grid_sample(x,F.affine_grid(theta,x.size(),align_corners=False),align_corners=False,padding_mode='zeros')

    def forward(self, x, params):
        phi=params['phi']; length=params['length']; width=params['sigma_x']
        x_rot=self._rotate(x,-phi)
        wl=self._soft_window(length,self.pos_long,length/2.0)
        ws=self._soft_window(width,self.pos_short,width)
        feat=(self._dynamic_depthwise(x_rot,self.conv_h,wl,True)+
              self._dynamic_depthwise(x_rot,self.conv_v,ws,False))
        feat=self._rotate(feat,phi)
        return x+self.pointwise(feat)


class GatedFusion(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.gate_fc=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),
            nn.Linear(in_ch,in_ch//2),nn.ReLU(inplace=True),nn.Linear(in_ch//2,1))

    def forward(self, psf_feat, strip_feat, identity):
        gate=torch.sigmoid(self.gate_fc(identity)).view(-1,1,1,1)
        return identity+gate*psf_feat+(1-gate)*strip_feat, gate.view(-1)


class DualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stage_stride=1):
        super().__init__()
        self.stage_stride=stage_stride
        self.psf=PSFChannel(in_ch); self.strip=StripChannel(in_ch)
        self.fusion=GatedFusion(in_ch)
        self.down=nn.Sequential(nn.Conv2d(in_ch,out_ch,3,stride=2,padding=1,bias=False),
                                nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))

    def forward(self, x, params):
        p=scale_params(params,self.stage_stride)
        psf_f=self.psf(x,p); strip_f=self.strip(x,p)
        fused,gate=self.fusion(psf_f,strip_f,x)
        return self.down(fused),gate


class SingleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_ch,in_ch,3,padding=1,bias=False),
                                nn.BatchNorm2d(in_ch),nn.ReLU(inplace=True))
        self.down=nn.Sequential(nn.Conv2d(in_ch,out_ch,3,stride=2,padding=1,bias=False),
                                nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))

    def forward(self, x):
        return self.down(x+self.conv(x))


class DualPathBackbone(nn.Module):
    def __init__(self, in_chans=1, stem_ch=32):
        super().__init__()
        self.stem=Stem(in_chans,stem_ch); self.estimator=GlobalParamEstimator(stem_ch)
        self.stage1=DualBlock(stem_ch,64,stage_stride=1)
        self.stage2=DualBlock(64,128,stage_stride=2)
        self.stage3=SingleBlock(128,256); self.stage4=SingleBlock(256,512)

    def forward(self, x):
        feat=self.stem(x); params=self.estimator(feat)
        f1,g1=self.stage1(feat,params); f2,g2=self.stage2(f1,params)
        f3=self.stage3(f2); f4=self.stage4(f3)
        return [f1,f2,f3,f4], params, [g1,g2]
