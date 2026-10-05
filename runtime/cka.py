"""Frozen encoder only. No segmentation model, decoder, optimizer or backward."""
import os,sys,json,time,hashlib,platform,argparse
from pathlib import Path
import numpy as np
import torch,timm,PIL
REPO=Path(__file__).resolve().parents[1]
R=Path(os.environ.get('CKA_OUTPUT',str(REPO/'runs/cka')))
B=REPO/'runtime'
sys.path.insert(0,str(B))
from data_io import Data

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def statehash(model):
 h=hashlib.sha256()
 for n,v in model.state_dict().items():h.update(n.encode());h.update(v.cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def cka(f):
 assert f.dtype==torch.float32 and f.ndim==3
 x=f-f.mean(dim=1,keepdim=True)
 selfnorm=torch.stack([(a.T@a).square().sum().sqrt() for a in x]);assert (selfnorm>0).all()
 out=torch.empty(12,12,dtype=torch.float32)
 for i in range(12):
  for j in range(i,12):
   v=(x[i].T@x[j]).square().sum()/(selfnorm[i]*selfnorm[j]);out[i,j]=v;out[j,i]=v
 assert torch.isfinite(out).all() and out.min()>=0 and out.max()<1.00001
 return out

def main():
 for folder in ('cache','audit','results'): (R/folder).mkdir(parents=True,exist_ok=True)
 ap=argparse.ArgumentParser();ap.add_argument('--sanity',action='store_true');a=ap.parse_args()
 torch.set_num_threads(4);torch.manual_seed(20261005);torch.use_deterministic_algorithms(True)
 cfg=json.loads((REPO/'configs/cka/protocol.json').read_text());assert sha(Path(os.environ['DINO_WEIGHTS']))==cfg['weight_sha256'];assert sha(REPO/'configs/cka/subset.jsonl')==cfg['subset_sha256']
 for n,h in json.loads((B/'public-freeze.json').read_text())['files'].items():assert sha(B/n)==h,n
 rows=[json.loads(s) for s in (REPO/'configs/cka/subset.jsonl').read_text().splitlines()];data=Data('train');ix={r['id']:i for i,r in enumerate(data.rows)}
 model=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=True,dynamic_img_size=True,pretrained_cfg_overlay={'file':os.environ['DINO_WEIGHTS'],'hf_hub_id':None}).cpu().eval().requires_grad_(False)
 assert not model.training and all(not p.requires_grad for p in model.parameters());before=statehash(model)
 mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
 records=[];matrices=[];valid_matrices=[];start=time.time()
 for k,row in enumerate(rows[:1] if a.sanity else rows):
  t=time.time();path=R/'cache'/f'{k:03d}.npy';metapath=path.with_suffix('.json')
  if path.exists():
   rec=json.loads(metapath.read_text());assert rec['sample_id']==row['id'] and sha(path)==rec['sha256'];f=torch.from_numpy(np.load(path)).float();meta=rec['geometry']
  else:
   x,_,_,meta=data.item(ix[row['id']],seed=None);assert x.shape==(3,448,448)
   with torch.inference_mode():features=model.get_intermediate_layers((x[None]-mean)/std,n=list(range(12)),reshape=False,norm=True)
   assert len(features)==12 and all(v.shape==(1,1024,384) and not v.requires_grad for v in features)
   original=torch.stack([v[0] for v in features]);assert torch.isfinite(original).all()
   cached=original.half().numpy();np.save(path,cached);f=torch.from_numpy(cached).float()
   rec=dict(sample_id=row['id'],split='train',center=row['center'],source=row['group'],kind=row['kind'],positive=row['reference_positive'],image_sha256=row['sha256'],feature_file=str(path.relative_to(R)),sha256=sha(path),bytes=path.stat().st_size,layers=list(range(1,13)),shape=list(cached.shape),dtype=str(cached.dtype),weight_sha256=cfg['weight_sha256'],protocol_sha256=sha(REPO/'configs/cka/protocol.json'),geometry=meta)
   write(metapath,rec)
  mat=cka(f)
  yy,xx=torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij');l=meta['left'];top=meta['top'];w=meta['resized_width'];h=meta['resized_height']
  valid=((xx*14>=l)&((xx+1)*14<=l+w)&(yy*14>=top)&((yy+1)*14<=top+h)).flatten();assert valid.sum()>384
  vm=cka(f[:,valid]);rec['fully_valid_patches']=int(valid.sum());records.append(rec);matrices.append(mat.numpy());valid_matrices.append(vm.numpy())
  if a.sanity:
   centered=f-f.mean(1,keepdim=True);g=centered[2]@centered[2].T;hgram=centered[8]@centered[8].T
   alternative=(g*hgram).sum()/(g.square().sum().sqrt()*hgram.square().sum().sqrt());gram_error=abs(float(alternative-mat[2,8]));assert gram_error<1e-5
   quantization_error=float((cka(original)-mat).abs().max()) if 'original' in locals() else None
   after=statehash(model);assert before==after
   write(R/'audit/sanity.json',dict(shape=list(f.shape),all_frozen=True,eval=True,no_grad=True,encoder_unchanged=before==after,state_sha256=before,feature_vs_gram_error=gram_error,fp16_cache_max_cka_error=quantization_error,seconds=time.time()-t,torch=torch.__version__,timm=timm.__version__,numpy=np.__version__,pillow=PIL.__version__,device='cpu',threads=4,fully_valid_patches=int(valid.sum())))
  print(json.dumps(dict(done=k+1,total=1 if a.sanity else 128,seconds=round(time.time()-t,2),elapsed=round(time.time()-start,2))),flush=True)
 if not a.sanity:
  after=statehash(model);assert before==after and all(p.grad is None for p in model.parameters())
  np.savez(R/'results/per-image-cka.npz',all_patches=np.stack(matrices),valid_patches=np.stack(valid_matrices),sample_ids=np.array([r['sample_id'] for r in records]))
  write(R/'cache/manifest.json',dict(protocol=cfg,samples=records));write(R/'audit/execution.json',dict(status='complete_feature_extraction_and_cka',n=len(records),device='cpu',torch=torch.__version__,timm=timm.__version__,numpy=np.__version__,pillow=PIL.__version__,platform=platform.platform(),encoder_state_before=before,encoder_state_after=after,encoder_unchanged=True,no_parameter_gradients=True,trainable_parameters=0,seconds=time.time()-start,code_sha256=sha(__file__),subset_sha256=sha(REPO/'configs/cka/subset.jsonl')))
if __name__=='__main__':main()
