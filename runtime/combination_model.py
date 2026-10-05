"""Phase 3: change only layer selection and grouping of fresh projection channels."""
import torch
from torch import nn
from torch.nn import functional as F
from models import DinoStudent

class CombinationStudent(DinoStudent):
    def __init__(self, layers, pretrained=True):
        self.layers = tuple(layers)
        assert self.layers in ((9,12),(6,9),(3,9))
        super().__init__(pretrained)
        original = self.projections
        with torch.random.fork_rng(devices=[]):
            projections = nn.ModuleList([nn.Conv2d(384,128,1) for _ in range(2)])
        with torch.no_grad():
            for i,p in enumerate(projections):
                p.weight.copy_(torch.cat([q.weight for q in original[2*i:2*i+2]],dim=0))
                p.bias.copy_(torch.cat([q.bias for q in original[2*i:2*i+2]],dim=0))
        self.projections = projections

    def forward(self,x):
        h,w=x.shape[-2:]
        assert h%14==0 and w%14==0
        with torch.no_grad():
            features=self.encoder.get_intermediate_layers(self.normalize(x),n=[v-1 for v in self.layers],reshape=True,norm=True)
        y=self.fuse(torch.cat([p(v) for p,v in zip(self.projections,features)],dim=1))
        y=F.interpolate(y,size=(h//4,w//4),mode='bilinear',align_corners=False)
        y=self.output(self.refine(y))
        return F.interpolate(y,size=(h,w),mode='bilinear',align_corners=False)
