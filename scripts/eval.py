"""Evaluate a trusted frozen, LastBlock or final LoRA checkpoint using the recorded native geometry."""
import argparse, csv, hashlib, json, os, sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'runtime'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,help='External JSONL manifest; omit for internal validation')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--model',choices=['frozen','lastblock','lora'],default='frozen')
    p.add_argument('--require-final-lock',action='store_true')
    a=p.parse_args()
    os.environ['POLYP_DATA_ROOT']=str(a.data_root.resolve())
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch, numpy as np
    from PIL import Image
    from models import DinoStudent
    from geometry import prepare,restore,score
    from rgb_view_rule import screen
    from left_view_rule import candidate
    from ablation_metrics import summary
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.require_final_lock:
        if a.model == 'lastblock': p.error('No published final lock for LastBlock')
        lock=json.loads((REPO/('configs/lora-final-model-lock.json' if a.model == 'lora' else 'configs/final-model-lock.json')).read_text())
        assert sha(a.checkpoint)==lock['checkpoint']['sha256'],'Wrong final checkpoint'
    from lastblock_model import LastBlockStudent
    from lora_model import LoRAStudent
    model={'frozen':DinoStudent,'lastblock':LastBlockStudent,'lora':LoRAStudent}[a.model](pretrained=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state.get('model',state),strict=True)
    model.requires_grad_(False);model.eval().to(a.device)
    a.output.mkdir(parents=True,exist_ok=False)
    if a.manifest:
        rows=[json.loads(s) for s in a.manifest.read_text().splitlines() if s]
        def item(i):
            r=rows[i];ip=a.data_root/r['image'];mp=a.data_root/r['mask']
            assert sha(ip)==r['image_sha256'] and sha(mp)==r['mask_sha256']
            rgb=Image.open(ip).convert('RGB');w,h=rgb.size
            rec=screen(rgb);rec.update(width=w,height=h);box=candidate(rec)
            x,_,_,meta=prepare(rgb.crop(box),None,448)
            meta.update(full_width=w,full_height=h,crop_box=box)
            return x,meta,np.asarray(Image.open(mp).convert('L'))
    else:
        from data_io import Data
        data=Data('validation');rows=data.rows
        def item(i):
            x,_,_,meta=data.item(i);_,mask=data.raw(i)
            return x,meta,np.asarray(mask)
    records=[]
    with torch.inference_mode():
        for i,r in enumerate(rows):
            x,meta,mask=item(i)
            pred=restore(model(x[None].to(a.device)),meta)[0,0].cpu().numpy()>0
            result=score(pred,mask>(127 if mask.max()>1 else 0))
            result.update(id=r['id'],group=r.get('group','external'),kind=r.get('kind','single'))
            records.append(result)
    result=summary(records,int(state.get('epoch',-1)))
    positives=[r for r in records if r['reference_positive']]
    result['positive_image_mean_iou']=float(np.mean([r['positive_iou'] for r in positives])) if positives else None
    from micro_metrics import aggregate
    result.update(aggregate(records))
    result['checkpoint_sha256']=sha(a.checkpoint)
    result['model']=a.model
    result['threshold_probability']=0.5
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (a.output/'per-image.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
