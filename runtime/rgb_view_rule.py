"""Read-only C1-C5 RGB layout screening, followed by reference-coverage checks.

No predictions/checkpoints are loaded. Candidate boxes are diagnostic, never applied
to the frozen training data. Heuristics are recorded before their first execution.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage


PREFIX = 'PolypGen2021_MultiCenterData_v3/'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def longest_run(a):
    best = 0
    for row in a:
        padded = np.r_[False, row, False].astype(np.int8)
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        if len(starts): best = max(best, int((ends-starts).max()))
    return best


def screen(rgb):
    w,h = rgb.size
    scale = min(512/w,512/h)
    tw,th = max(1,round(w*scale)),max(1,round(h*scale))
    a = np.asarray(rgb.resize((tw,th),Image.Resampling.BILINEAR))
    support = ndimage.binary_closing(a.max(axis=2)>40, structure=np.ones((3,3)))
    labels,n = ndimage.label(support)
    counts = np.bincount(labels.ravel());counts[0] = 0
    largest = int(counts.argmax()) if n else 0
    if not largest:
        return dict(box=[0,0,w,h],candidate_status='NO_COMPONENT',screen_outside_fraction=0,
                    box_fraction=1,other_components=0,largest_component_fraction=0,green_rectangles=[])
    ys,xs = np.nonzero(labels==largest)
    pad=max(1,math.ceil(.01*max(tw,th)))
    left,top=max(0,int(xs.min())-pad),max(0,int(ys.min())-pad)
    right,bottom=min(tw,int(xs.max())+pad+1),min(th,int(ys.max())+pad+1)
    outside=support.copy();outside[top:bottom,left:right]=False
    box=[max(0,math.floor(left*w/tw)),max(0,math.floor(top*h/th)),min(w,math.ceil(right*w/tw)),min(h,math.ceil(bottom*h/th))]
    # Saturated green rectilinear marks: flags only, never a semantic annotation label.
    ai=a.astype(np.int16)
    green=(ai[:,:,1]>150)&(ai[:,:,1]-ai[:,:,0]>50)&(ai[:,:,1]-ai[:,:,2]>50)
    gl,gn=ndimage.label(ndimage.binary_closing(green,structure=np.ones((2,2))))
    rectangles=[]
    for label,sl in enumerate(ndimage.find_objects(gl),1):
        if sl is None:continue
        sy,sx=sl;gh,gw=sy.stop-sy.start,sx.stop-sx.start
        if min(gh,gw)<.04*max(tw,th):continue
        component=gl[sl]==label
        density=float(component.mean())
        if density>.4:continue
        hr,vr=longest_run(component),longest_run(component.T)
        if hr<.5*gw or vr<.5*gh:continue
        cx,cy=(sx.start+sx.stop)/2,(sy.start+sy.stop)/2
        rectangles.append(dict(box=[int(sx.start*w/tw),int(sy.start*h/th),int(sx.stop*w/tw),int(sy.stop*h/th)],
                               density=density,horizontal_run=hr,vertical_run=vr,
                               inside_candidate_view=bool(left<=cx<right and top<=cy<bottom)))
    return dict(box=box,candidate_status='CANDIDATE' if counts[largest]/support.size>=.2 else 'SMALL_COMPONENT_UNCERTAIN',
                screen_outside_fraction=float(outside.sum()/support.size),
                box_fraction=(box[2]-box[0])*(box[3]-box[1])/(w*h),
                other_components=int(sum(x>=.003*support.size for i,x in enumerate(counts) if i!=largest)),
                largest_component_fraction=float(counts[largest]/support.size),green_rectangles=rectangles)
