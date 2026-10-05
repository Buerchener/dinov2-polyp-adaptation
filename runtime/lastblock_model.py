"""Identical four-layer Fusion architecture; only last-block gradient path changes."""
import torch
from torch import nn
from torch.nn import functional as F
from models import DinoStudent

class LastBlockStudent(DinoStudent):
    def __init__(self,pretrained=True):
        super().__init__(pretrained)
        assert len(self.encoder.blocks)==12
        self.encoder.blocks[11].requires_grad_(True)

    def train(self,mode=True):
        nn.Module.train(self,mode)
        # Frozen components remain eval. Do not disable autograd around encoder:
        # frozen inputs/weights naturally have no graph until trainable block 12.
        for module in self.encoder.modules(): module.training=False
        self.encoder.training=mode
        self.encoder.blocks.training=mode
        self.encoder.blocks[11].train(mode)
        return self
    def forward(self,x):
        h,w=x.shape[-2:]
        assert h%14==0 and w%14==0,'DINO inputs must respect the patch grid'
        features=self.encoder.get_intermediate_layers(self.normalize(x),n=[2,5,8,11],reshape=True,norm=True)
        y=self.fuse(torch.cat([p(v) for p,v in zip(self.projections,features)],dim=1))
        y=F.interpolate(y,size=(h//4,w//4),mode='bilinear',align_corners=False)
        y=self.output(self.refine(y))
        return F.interpolate(y,size=(h,w),mode='bilinear',align_corners=False)


def parameter_groups(model):
    head=[p for n,p in model.named_parameters() if not n.startswith('encoder.') and p.requires_grad]
    last=list(model.encoder.blocks[11].parameters())
    allowed={id(p) for p in head+last}
    assert allowed=={id(p) for p in model.parameters() if p.requires_grad}
    assert all(not p.requires_grad for n,p in model.encoder.named_parameters() if not n.startswith('blocks.11.'))
    return [dict(params=head,lr=1e-3,name='head'),dict(params=last,lr=1e-5,name='last_block')]
