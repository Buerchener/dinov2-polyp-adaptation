"""Positive-only global Dice; normal frames are evaluated separately."""
import math


def aggregate(records):
    if not records:
        raise ValueError('No records')
    seen=set(); positives=[]; normals=[]
    for r in records:
        if r['id'] in seen: raise ValueError('Duplicate image ID')
        seen.add(r['id'])
        tp,pp,gt,n=(r[k] for k in ('intersection','prediction_pixels','gt_pixels','image_pixels'))
        if any(isinstance(x,bool) or not isinstance(x,int) for x in (tp,pp,gt,n)):
            raise ValueError('Pixel counts must be integers')
        if not (n>0 and 0<=tp<=min(pp,gt) and max(pp,gt)<=n and pp+gt-tp<=n):
            raise ValueError('Invalid pixel counts')
        if type(r['reference_positive']) is not bool or r['reference_positive'] != (gt>0):
            raise ValueError('Inconsistent positive flag')
        (positives if gt else normals).append(r)
    tp=sum(r['intersection'] for r in positives)
    fp=sum(r['prediction_pixels']-r['intersection'] for r in positives)
    fn=sum(r['gt_pixels']-r['intersection'] for r in positives)
    return dict(positive_n=len(positives),normal_n=len(normals),TP=tp,FP=fp,FN=fn,
        positive_micro_dice=2*tp/(2*tp+fp+fn) if positives else None,
        positive_macro_dice=math.fsum(2*r['intersection']/(r['prediction_pixels']+r['gt_pixels']) for r in positives)/len(positives) if positives else None,
        normal_any_pixel_fp=sum(r['prediction_pixels']>0 for r in normals),
        normal_area_over_0p1pct_fp=sum(r['prediction_pixels']/r['image_pixels']>.001 for r in normals),
        normal_mean_foreground=math.fsum(r['prediction_pixels']/r['image_pixels'] for r in normals)/len(normals) if normals else None,
        empty_positive_predictions=sum(r['prediction_pixels']==0 for r in positives))
