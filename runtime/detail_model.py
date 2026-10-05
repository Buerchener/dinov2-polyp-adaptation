"""One RGB detail path and one skip at 112x112; unchanged Fusion core."""
import torch
from torch import nn
from torch.nn import functional as F
from models import DinoStudent

class DetailStudent(DinoStudent):
    def __init__(self, pretrained=True):
        super().__init__(pretrained)
        # Preserve baseline initialization and post-construction RNG trajectory.
        with torch.random.fork_rng(devices=[]):
            self.detail=nn.Sequential(
                nn.Conv2d(3,32,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,32),nn.GELU(),
                nn.Conv2d(32,64,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,64),nn.GELU())
            self.detail_merge=nn.Conv2d(192,128,3,padding=1,bias=False)

    def forward(self,x):
        h,w=x.shape[-2:]
        assert (h,w)==(448,448)
        with torch.no_grad():
            features=self.encoder.get_intermediate_layers(self.normalize(x),n=[2,5,8,11],reshape=True,norm=True)
        y=self.fuse(torch.cat([p(v) for p,v in zip(self.projections,features)],dim=1))
        y=F.interpolate(y,size=(h//4,w//4),mode='bilinear',align_corners=False)
        detail=self.detail(x)
        y=self.detail_merge(torch.cat([y,detail],dim=1))
        y=self.output(self.refine(y))
        return F.interpolate(y,size=(h,w),mode='bilinear',align_corners=False)
