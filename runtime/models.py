"""Two explicit, local-weight student candidates. No network calls."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import timm
import os

ROOT=Path(__file__).resolve().parents[1]


class Normalized(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
    def normalize(self,x):return (x-self.mean)/self.std



def conv(in_channels,out_channels):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,3,padding=1,bias=False),nn.GroupNorm(8,out_channels),nn.GELU())


class DinoStudent(Normalized):
    """Frozen ViT-S/14, four intermediate features, fixed small segmentation head."""
    def __init__(self,pretrained=True):
        super().__init__()
        kwargs={}
        if pretrained:kwargs['pretrained_cfg_overlay']={'file':os.environ['DINO_WEIGHTS'],'hf_hub_id':None}
        self.encoder=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=pretrained,dynamic_img_size=True,**kwargs)
        self.encoder.requires_grad_(False);self.encoder.eval()
        self.projections=nn.ModuleList([nn.Conv2d(384,64,1) for _ in range(4)])
        self.fuse=conv(256,128);self.refine=conv(128,64);self.output=nn.Conv2d(64,1,1)
    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
    def forward(self,x):
        h,w=x.shape[-2:]
        assert h%14==0 and w%14==0,'DINO inputs must respect the patch grid'
        with torch.no_grad():features=self.encoder.get_intermediate_layers(self.normalize(x),n=[2,5,8,11],reshape=True,norm=True)
        y=self.fuse(torch.cat([p(v) for p,v in zip(self.projections,features)],dim=1))
        y=F.interpolate(y,size=(h//4,w//4),mode='bilinear',align_corners=False)
        y=self.output(self.refine(y))
        return F.interpolate(y,size=(h,w),mode='bilinear',align_corners=False)


def counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
