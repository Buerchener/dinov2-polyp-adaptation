"""Shared aspect-preserving input geometry and positive/negative metrics."""
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

MEAN=torch.tensor([.485,.456,.406])[:,None,None]


def prepare(rgb,mask=None,side=448):
    assert isinstance(rgb,Image.Image)
    rgb=rgb.convert('RGB');w,h=rgb.size;scale=min(side/w,side/h)
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale));left,top=(side-nw)//2,(side-nh)//2
    x=MEAN.expand(3,side,side).clone();valid=torch.zeros(1,side,side);y=torch.zeros_like(valid)
    scaled=rgb.resize((nw,nh),Image.Resampling.BILINEAR)
    x[:,top:top+nh,left:left+nw]=torch.from_numpy(np.asarray(scaled).copy()).permute(2,0,1).float()/255
    valid[:,top:top+nh,left:left+nw]=1
    if mask is not None:
        raw=np.asarray(mask.convert('L'));assert raw.shape==(h,w)
        binary=Image.fromarray((raw>(127 if raw.max()>1 else 0)).astype('uint8'))
        y[:,top:top+nh,left:left+nw]=torch.from_numpy(np.asarray(binary.resize((nw,nh),Image.Resampling.NEAREST)).copy()).float()
    return x,y,valid,dict(width=w,height=h,resized_width=nw,resized_height=nh,left=left,top=top,side=side)


def restore(logits,meta):
    assert logits.ndim==4 and logits.shape[:2]==(1,1)
    t,l=meta['top'],meta['left'];nh,nw=meta['resized_height'],meta['resized_width']
    patch=F.interpolate(logits[:,:,t:t+nh,l:l+nw],size=(meta['height'],meta['width']),mode='bilinear',align_corners=False)
    if 'crop_box' not in meta:return patch
    left,top,right,bottom=meta['crop_box']
    assert (right-left,bottom-top)==(meta['width'],meta['height'])
    full=patch.new_full((1,1,meta['full_height'],meta['full_width']),-30.)
    full[:,:,top:bottom,left:right]=patch
    return full


def loss_fn(logits,truth,valid):
    assert logits.shape==truth.shape==valid.shape and (valid.sum((1,2,3))>0).all()
    axes=(1,2,3)
    bce=(F.binary_cross_entropy_with_logits(logits,truth,reduction='none')*valid).sum(axes)/valid.sum(axes)
    p=logits.sigmoid()*valid;g=truth*valid
    dice=(2*(p*g).sum(axes)+1)/(p.sum(axes)+g.sum(axes)+1)
    return (bce+1-dice).mean()


def score(pred,truth):
    p=np.asarray(pred,dtype=bool);g=np.asarray(truth,dtype=bool);assert p.shape==g.shape
    inter=int((p&g).sum());pp=int(p.sum());gg=int(g.sum());area=p.size
    return dict(reference_positive=gg>0,intersection=inter,prediction_pixels=pp,gt_pixels=gg,image_pixels=area,
                positive_dice=2*inter/(pp+gg) if gg else None,positive_iou=inter/(pp+gg-inter) if gg else None,
                severe=(2*inter/(pp+gg)<.5) if gg else None,
                negative_any_false_positive=bool(pp>0) if not gg else None,
                negative_fp_fraction=pp/area if not gg else None,
                negative_fp_over_0p1pct=bool(pp/area>.001) if not gg else None)
