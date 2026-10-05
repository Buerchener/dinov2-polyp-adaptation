"""Offline CPU checks; no pretrained download, patient data or GPU needed."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from models import DinoStudent, counts
from geometry import prepare, restore, score, loss_fn
from source_sampler import draws
from data_io import Data
from ablation_model import SingleLayerStudent
from combination_model import CombinationStudent
from detail_model import DetailStudent
from lastblock_model import LastBlockStudent
from hierarchical_model import HierarchicalStudent
from cka import cka

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    for rel,digest in json.loads((ROOT/'runtime/public-freeze.json').read_text())['files'].items():
        assert hashlib.sha256((ROOT/'runtime'/rel).read_bytes()).hexdigest()==digest,rel
    train,val=Data('train'),Data('validation')
    assert (len(train.rows),len(val.rows))==(1066,257)
    assert sum(not r.get('normal_expansion_candidate') for r in train.rows)==970
    h=hashlib.sha256()
    for ids,seeds in draws(train.rows,20261002,3040,'old_C1'):
        assert len(ids)==9 and not any(train.rows[i].get('normal_expansion_candidate') for i in ids)
        h.update(json.dumps([ids,seeds],separators=(',',':')).encode())
    assert h.hexdigest()==json.loads((ROOT/'configs/frozen.json').read_text())['draw_sha256']
    model=DinoStudent(pretrained=False);model.train()
    assert counts(model)==dict(total=22523841,trainable=467649)
    assert not model.encoder.training and not any(p.requires_grad for p in model.encoder.parameters())
    x,y,v,meta=prepare(Image.new('RGB',(87,51),(120,90,80)),Image.new('L',(87,51),255))
    logits=model(x[None]);assert logits.shape==(1,1,448,448)
    loss=loss_fn(logits,y[None],v[None]);loss.backward()
    assert torch.isfinite(loss) and model.output.weight.grad is not None
    assert all(p.grad is None for p in model.encoder.parameters())
    restored=restore(torch.ones_like(logits),meta)
    assert restored.shape==(1,1,51,87)
    assert score(restored.detach().numpy()[0,0]>0,np.ones((51,87)))['positive_dice']==1
    assert score(np.zeros((4,4)),np.zeros((4,4)))['positive_dice'] is None
    assert score(np.eye(4),np.zeros((4,4)))['negative_any_false_positive']
    f=torch.randn(12,32,16);matrix=cka(f)
    assert torch.allclose(matrix,matrix.T) and torch.allclose(matrix.diag(),torch.ones(12),atol=1e-5)
    for create,expected in [(lambda:SingleLayerStudent(12,False),467649),(lambda:CombinationStudent((9,12),False),467649),(lambda:DetailStudent(False),708321),(lambda:LastBlockStudent(False),2242881),(lambda:HierarchicalStudent(False),478465)]:
        variant=create();assert counts(variant)['trainable']==expected
        variant.eval()
        with torch.no_grad():assert variant(x[None]).shape==(1,1,448,448)
        del variant
    print(json.dumps(dict(status='PASS',device='cpu',network=False,gpu_training=False,checks=['public source hashes','ordered split and 3040-step sampler hash','all model parameter counts and forward shapes','frozen gradient boundary','native restore and positive/normal metrics','CKA symmetry/diagonal'])))
if __name__=='__main__':main()
