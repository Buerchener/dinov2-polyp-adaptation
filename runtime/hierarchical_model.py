"""Single parameter-matched staged fusion; all native ViT grids are 32x32."""
import torch
from torch import nn
from torch.nn import functional as F
from models import DinoStudent, conv


class HierarchicalStudent(DinoStudent):
    def __init__(self, pretrained=True):
        super().__init__(pretrained)
        # Remove the entire original head; only the frozen encoder is reused.
        del self.projections, self.fuse, self.refine, self.output
        self.projections = nn.ModuleList([nn.Conv2d(384, c, 1) for c in (32, 64, 96, 128)])
        self.stage1 = conv(128 + 96, 112)
        self.stage2 = conv(112 + 64, 64)
        self.stage3 = conv(64 + 32, 32)
        self.output = nn.Conv2d(32, 1, 1)

    @staticmethod
    def resize(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        assert (h, w) == (448, 448)
        assert not self.encoder.training
        with torch.no_grad():
            features = self.encoder.get_intermediate_layers(
                self.normalize(x), n=[2, 5, 8, 11], reshape=True, norm=True)
        l3, l6, l9, l12 = [p(f) for p, f in zip(self.projections, features)]
        assert all(f.shape[-2:] == (32, 32) for f in features)
        y = self.stage1(torch.cat((l12, l9), dim=1))
        y = self.stage2(torch.cat((self.resize(y, (64, 64)), self.resize(l6, (64, 64))), dim=1))
        y = self.stage3(torch.cat((self.resize(y, (112, 112)), self.resize(l3, (112, 112))), dim=1))
        return self.resize(self.output(y), (h, w))
