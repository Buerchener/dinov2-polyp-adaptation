"""Recompute micro/macro Dice from integer counts or native binary prediction masks."""
import argparse, hashlib, io, json, sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from micro_metrics import aggregate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--records',type=Path,help='JSONL with image IDs and integer pixel counts')
    mode.add_argument('--predictions',type=Path,help='ZIP of binary native masks at <manifest id>.png')
    p.add_argument('--data-root',type=Path)
    p.add_argument('--manifest',type=Path,default=ROOT/'runtime/reports/manifest.jsonl')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.records:
        records=[json.loads(s) for s in a.records.read_text().splitlines() if s]
    else:
        if a.data_root is None: p.error('--predictions requires --data-root')
        import numpy as np
        from PIL import Image
        rows=[json.loads(s) for s in a.manifest.read_text().splitlines() if s]
        rows=[r for r in rows if r['role']=='validation']
        if len({r['id'] for r in rows}) != len(rows): raise ValueError('Duplicate manifest IDs')
        records=[]
        with zipfile.ZipFile(a.predictions) as archive:
            expected={r['id']+'.png' for r in rows}
            if set(archive.namelist())!=expected or len(archive.namelist())!=len(expected):
                raise ValueError('Prediction archive does not match validation IDs exactly')
            for r in rows:
                pred=np.asarray(Image.open(io.BytesIO(archive.read(r['id']+'.png'))).convert('L'))
                if not set(np.unique(pred)).issubset({0,1,255}): raise ValueError('Expected binary masks, not logits')
                pred=pred>0
                if r['mask']:
                    raw=(a.data_root/r['mask']).read_bytes()
                    if hashlib.sha256(raw).hexdigest()!=r['mask_sha256']: raise ValueError('GT hash mismatch')
                    gt=np.asarray(Image.open(io.BytesIO(raw)).convert('L'))>127
                else: gt=np.zeros((r['height'],r['width']),dtype=bool)
                if pred.shape!=gt.shape: raise ValueError('Native shape mismatch')
                if bool(gt.any())!=r['reference_positive']: raise ValueError('GT flag mismatch')
                records.append(dict(id=r['id'],reference_positive=bool(gt.any()),intersection=int((pred&gt).sum()),prediction_pixels=int(pred.sum()),gt_pixels=int(gt.sum()),image_pixels=int(pred.size)))
    result=aggregate(records)
    result['mode']='saved_integer_counts' if a.records else 'native_binary_masks_vs_hash_verified_GT'
    text=json.dumps(result,indent=2)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text,end='')

if __name__=='__main__':main()
