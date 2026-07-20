"""
v1 Backbone: DualPathV1 — 每 block 自带 GlobalParamEstimator（σ/θ/φ/L + gate）
==============================================================================
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
    def __init__(self, in_ch, hidden=64, attn_size=32, num_heads=4):
        super().__init__()
        self.attn_size = attn_size; self.hidden = hidden
        self.reduce = nn.Conv2d(in_ch, hidden, 1)
        self.attn   = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.norm   = nn.LayerNorm(hidden)
        self.fc_psf    = nn.Linear(hidden, 3)
        self.fc_streak = nn.Linear(hidden, 2)

    def forward(self, x):
        B = x.shape[0]
        h = self.reduce(x); h = F.adaptive_avg_pool2d(h, self.attn_size)
        seq = h.flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(seq, seq, seq); seq = self.norm(seq + attn_out)
        gf = seq.mean(dim=1)
        psf_raw = self.fc_psf(gf); streak_raw = self.fc_streak(gf)
        sigma_x = F.softplus(psf_raw[:,0])+0.5; sigma_y = F.softplus(psf_raw[:,1])+0.5
        theta   = torch.tanh(psf_raw[:,2])*(math.pi/2)
        phi     = torch.tanh(streak_raw[:,0])*(math.pi/2)
        length  = F.softplus(streak_raw[:,1])+1.0
        return {'sigma_x':sigma_x,'sigma_y':sigma_y,'theta':theta,'phi':phi,'length':length}, gf


class PSFChannel(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.ch=channels; self.ks=kernel_size
        self.pointwise=nn.Conv2d(channels,channels,1)

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
    def __init__(self, channels, k_long=15, k_short=5):
        super().__init__()
        self.ch=channels
        self.conv_h=nn.Conv2d(channels,channels,(1,k_long),padding=(0,k_long//2),groups=channels,bias=False)
        self.conv_v=nn.Conv2d(channels,channels,(k_short,1),padding=(k_short//2,0),groups=channels,bias=False)
        self.pointwise=nn.Conv2d(channels,channels,1)

    def _rotate(self, x, angle):
        B,C,H,W=x.shape; cos,sin=angle.cos(),angle.sin()
        theta=torch.zeros(B,2,3,device=x.device,dtype=x.dtype)
        theta[:,0,0]=cos;theta[:,0,1]=-sin;theta[:,1,0]=sin;theta[:,1,1]=cos
        return F.grid_sample(x,F.affine_grid(theta,x.size(),align_corners=False),align_corners=False,padding_mode='zeros')

    def forward(self, x, params):
        phi=params['phi']; x_rot=self._rotate(x,-phi)
        feat=self.conv_h(x_rot)+self.conv_v(x_rot)
        return x+self.pointwise(self._rotate(feat,phi))


class GatedFusion(nn.Module):
    def __init__(self, global_dim):
        super().__init__()
        self.gate_fc=nn.Sequential(nn.Linear(global_dim,global_dim//2),nn.ReLU(inplace=True),nn.Linear(global_dim//2,1))

    def forward(self, psf_feat, strip_feat, identity, global_feat):
        gate=torch.sigmoid(self.gate_fc(global_feat)).view(-1,1,1,1)
        return identity+gate*psf_feat+(1-gate)*strip_feat, gate.view(-1)


class DualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, global_hidden=64):
        super().__init__()
        self.estimator=GlobalParamEstimator(in_ch,hidden=global_hidden)
        self.psf=PSFChannel(in_ch); self.strip=StripChannel(in_ch)
        self.fusion=GatedFusion(global_hidden)
        self.down=nn.Sequential(nn.Conv2d(in_ch,out_ch,3,stride=2,padding=1,bias=False),
                                nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))

    def forward(self, x):
        params,gf=self.estimator(x)
        psf_f=self.psf(x,params); strip_f=self.strip(x,params)
        fused,gate=self.fusion(psf_f,strip_f,x,gf)
        return self.down(fused),params,gate


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
        self.stem=Stem(in_chans,stem_ch)
        self.stage1=DualBlock(stem_ch,64); self.stage2=DualBlock(64,128)
        self.stage3=SingleBlock(128,256); self.stage4=SingleBlock(256,512)

    def forward(self, x):
        feat=self.stem(x)
        f1,p1,g1=self.stage1(feat); f2,p2,g2=self.stage2(f1)
        f3=self.stage3(f2); f4=self.stage4(f3)
        return [f1,f2,f3,f4], [p1,p2], [g1,g2]
