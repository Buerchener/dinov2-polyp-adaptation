"""All 12 fused qkv projections, r4/alpha8; original EarlyFusion unchanged."""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from models import DinoStudent


class LoRAQKV(nn.Module):
    def __init__(self, base):
        super().__init__()
        assert isinstance(base, nn.Linear)
        assert (base.in_features, base.out_features) == (384, 1152)
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(4, 384))
        self.lora_B = nn.Parameter(torch.zeros(1152, 4))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = 8 / 4

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


class LoRAStudent(DinoStudent):
    def __init__(self, pretrained=True):
        super().__init__(pretrained)
        assert len(self.encoder.blocks) == 12
        # Constructor randomness never changes the seeded decoder or training RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261002 + 400)
            for block in self.encoder.blocks:
                block.attn.qkv = LoRAQKV(block.attn.qkv)

    def forward(self, x):
        h, w = x.shape[-2:]
        assert h % 14 == 0 and w % 14 == 0
        features = self.encoder.get_intermediate_layers(
            self.normalize(x), n=[2, 5, 8, 11], reshape=True, norm=True)
        y = self.fuse(torch.cat([p(v) for p, v in zip(self.projections, features)], dim=1))
        y = F.interpolate(y, size=(h // 4, w // 4), mode='bilinear', align_corners=False)
        y = self.output(self.refine(y))
        return F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)


def is_lora(name):
    return name.endswith('.lora_A') or name.endswith('.lora_B')


def tensor_digest(items):
    h = hashlib.sha256()
    for name, value in items:
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def encoder_digest(model):
    # Normalize wrapped base keys to the original encoder namespace.
    return tensor_digest((n.replace('.qkv.base.', '.qkv.'), v)
                         for n, v in model.encoder.state_dict().items() if not is_lora(n))


def decoder_digest(model):
    return tensor_digest((n, v) for n, v in model.state_dict().items() if not n.startswith('encoder.'))


def parameter_groups(model):
    head = [p for n, p in model.named_parameters() if not n.startswith('encoder.') and p.requires_grad]
    lora = [p for n, p in model.named_parameters() if is_lora(n) and p.requires_grad]
    assert sum(p.numel() for p in head) == 467649
    expected = 73728 if isinstance(model, LoRAStudent) else 0
    assert sum(p.numel() for p in lora) == expected
    allowed = {id(p) for p in head + lora}
    assert allowed == {id(p) for p in model.parameters() if p.requires_grad}
    assert all(not p.requires_grad for n, p in model.encoder.named_parameters() if not is_lora(n))
    groups = [dict(params=head, lr=1e-3, name='decoder')]
    if lora:
        groups.append(dict(params=lora, lr=5e-5, name='lora'))
    return groups


def gradient_audit(model):
    trainable = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
            trainable.append(n)
        else:
            assert p.grad is None, n
    return dict(only_decoder_and_lora_have_gradients=True, connected_trainable_tensors=trainable,
                zero_A_gradient_at_initial_zero_B_is_expected=True)
