"""Only feature selection/projection differ from the frozen old_C1 model."""
import torch
from torch import nn
from torch.nn import functional as F
from models import DinoStudent

class SingleLayerStudent(DinoStudent):
    def __init__(self, layer, pretrained=True):
        assert layer in (3,6,9,12)
        self.layer = layer
        # Recreate exactly the original fresh initialization, never trained weights.
        super().__init__(pretrained)
        original = self.projections
        # Construction must not advance RNG beyond the baseline initialization.
        with torch.random.fork_rng(devices=[]):
            projection = nn.Conv2d(384, 256, 1)
        with torch.no_grad():
            projection.weight.copy_(torch.cat([p.weight for p in original], dim=0))
            projection.bias.copy_(torch.cat([p.bias for p in original], dim=0))
        self.projections = nn.ModuleList([projection])

    def forward(self, x):
        h, w = x.shape[-2:]
        assert h % 14 == 0 and w % 14 == 0
        with torch.no_grad():
            features = self.encoder.get_intermediate_layers(
                self.normalize(x), n=[self.layer-1], reshape=True, norm=True)
        y = self.fuse(self.projections[0](features[0]))
        y = F.interpolate(y, size=(h//4, w//4), mode='bilinear', align_corners=False)
        y = self.output(self.refine(y))
        return F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
