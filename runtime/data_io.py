"""Explicit role whitelist; confirmation and reserve images cannot be loaded here."""
import os
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image,ImageEnhance
import torch
from geometry import prepare

ROOT=Path(__file__).resolve().parent
DATA_ROOT=Path(os.environ.get("POLYP_DATA_ROOT", str(ROOT.parent / "datasets")))
PREFIX='PolypGen2021_MultiCenterData_v3/'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


class Data:
    def __init__(self,role):
        assert role in ('train','validation'),'Sealed/reserve cohort access is prohibited by this development loader'
        path=ROOT/'reports/manifest.jsonl';summary=json.loads((ROOT/'reports/audit-summary.json').read_text())
        assert digest(path)==summary['manifest_sha256']
        self.rows=[json.loads(x) for x in path.read_text().splitlines() if x]
        self.rows=[r for r in self.rows if r['role']==role];self.role=role;self.archive=None
        assert self.rows and all(r['mask_audited'] for r in self.rows)
        self.crop_boxes=json.loads((ROOT/'crop-boxes.json').read_text())
        assert all(r['id'] in self.crop_boxes for r in self.rows)
    def read(self,rel,expected=None):
        local=DATA_ROOT/rel
        if local.exists():raw=local.read_bytes()
        else:
            if self.archive is None:self.archive=zipfile.ZipFile(DATA_ROOT/'PolypGen2021_MultiCenterData_v3.zip')
            raw=self.archive.read(PREFIX+rel)
        if expected:assert hashlib.sha256(raw).hexdigest()==expected,rel
        return Image.open(io.BytesIO(raw)).copy()
    def raw(self,index):
        r=self.rows[index];rgb=self.read(r['image'],r['sha256']).convert('RGB')
        mask=self.read(r['mask'],r['mask_sha256']).convert('L') if r['mask'] else Image.new('L',rgb.size)
        return rgb,mask
    def item(self,index,seed=None):
        rgb,mask=self.raw(index)
        width,height=rgb.size;box=self.crop_boxes[self.rows[index]['id']]
        l,t,r,b=box;assert 0<=l<r<=width and 0<=t<b<=height
        rgb=rgb.crop(box);mask=mask.crop(box)
        if seed is not None:
            rng=np.random.default_rng(seed)
            if rng.random()<.5:rgb=rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT);mask=mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if rng.random()<.5:rgb=rgb.transpose(Image.Transpose.FLIP_TOP_BOTTOM);mask=mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            rgb=ImageEnhance.Brightness(rgb).enhance(float(rng.uniform(.9,1.1)))
            rgb=ImageEnhance.Contrast(rgb).enhance(float(rng.uniform(.9,1.1)))
        x,y,valid,meta=prepare(rgb,mask,448)
        meta.update(full_width=width,full_height=height,crop_box=box)
        return x,y,valid,meta


def balanced_draws(rows,seed,steps):
    rng=np.random.default_rng(seed);centers=['C1','C2','C3','C4']
    groups={c:[i for i,r in enumerate(rows) if r['center']==c and not r.get('normal_expansion_candidate',False)] for c in centers}
    added=[i for i,r in enumerate(rows) if r.get('normal_expansion_candidate',False)]
    assert not added or len(added)==111
    replacement=np.random.default_rng(seed+123000)

    assert all(len(v)>=2 for v in groups.values())
    for _ in range(steps):
        ids=[int(i) for c in centers for i in rng.choice(groups[c],2,replace=False)]
        aug=[int(s) for s in rng.integers(0,2**31-1,len(ids))]
        used=set()
        for j,i in enumerate(ids):
            if added and not rows[i]['reference_positive'] and replacement.random()<111/228:
                pool=[k for k in added if k not in used]
                ids[j]=int(replacement.choice(pool));used.add(ids[j])
        yield ids,aug


def batch(data,ids,seeds=None):
    values=[data.item(i,None if seeds is None else seeds[j]) for j,i in enumerate(ids)]
    return tuple(torch.stack([v[k] for v in values]) for k in range(3))
