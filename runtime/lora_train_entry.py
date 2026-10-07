"""Matched Frozen/LoRA replay. Fixed epoch20, no decoder changes."""
import hashlib
import io
import json
import random
import subprocess
import sys
import time
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

EXPECTED_DRAW='457640139e657d7307b3f2124bf2dd78f25a1bc3eebf212d8a44773561e696e8'
PACKAGES={'timm':'1.0.26','segmentation-models-pytorch':'0.5.0','numpy':'2.1.3','Pillow':'11.3.0'}

def run(ctx):
    import torch
    from core import atomic_json
    P=Path(__file__).resolve().parents[1]
    assert torch.cuda.is_available(), 'Training requires CUDA'
    import numpy as np
    from PIL import Image
    from full_state import capture, restore as restore_state
    root=Path(__file__).resolve().parent
    sys.path[:0]=[str(root/'runtime'),str(root)]
    from data_io import Data,batch
    from geometry import loss_fn,restore,score
    from source_sampler import draws
    from lora_model import LoRAStudent,parameter_groups,encoder_digest,decoder_digest,gradient_audit
    from ablation_metrics import summary
    for name,h in json.loads((root/'public-freeze.json').read_text())['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    assert ctx.config['target_epoch']==20 and ctx.config['required_evaluations']==[0,5,10,15,20]
    seed=20261002
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    train,val=Data('train'),Data('validation')
    assert len(train.rows)==1066 and len(val.rows)==257
    assert all(r['center'] in ('C1','C2','C3','C4') for r in train.rows+val.rows)
    replay=list(draws(train.rows,seed,3040,'old_C1'))
    trajectory=hashlib.sha256();prefix=[trajectory.hexdigest()]
    for ids,seeds in replay:
        assert len(ids)==9 and all(not train.rows[i].get('normal_expansion_candidate') for i in ids)
        trajectory.update(json.dumps([ids,seeds],separators=(',',':')).encode());prefix.append(trajectory.hexdigest())
    assert prefix[-1]==EXPECTED_DRAW
    from models import DinoStudent
    assert ctx.config['arm'] == 'C_LoRA_LR5e5'
    model=LoRAStudent().to('cuda')
    initial_encoder_sha256=encoder_digest(model)
    init=dict(encoder_sha256=initial_encoder_sha256,decoder_sha256=decoder_digest(model),
              cpu_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
              cuda_rng_sha256=[hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in torch.cuda.get_rng_state_all()])
    common=P/'configs/lora-initialization.json'
    expected=json.loads(common.read_text())
    assert all(expected[k]==init[k] for k in ('encoder_sha256','decoder_sha256')), 'Initial weights differ from historical experiment'
    init['historical_rng_match'] = all(expected[k]==init[k] for k in ('cpu_rng_sha256','cuda_rng_sha256'))
    atomic_json(ctx.work/'initialization.json',init)
    torch.cuda.reset_peak_memory_stats()
    head=[p for p in model.parameters() if p.requires_grad]
    groups=parameter_groups(model)
    assert sum(p.numel() for p in groups[0]['params'])==467649
    assert [g['lr'] for g in groups] == [1e-3, 5e-5]
    atomic_json(ctx.work/'optimizer-lr-audit.json',dict(decoder_lr=groups[0]['lr'],lora_lr=groups[1]['lr']))
    optimizer=torch.optim.AdamW(groups,weight_decay=.0001)
    epoch,step,updates,evaluations,training_log=0,0,0,{},[]
    state=ctx.load()
    if state:
        restore_state(state,model,optimizer)
        epoch,step,updates=(state['progress'][k] for k in ('epoch','step','optimizer_updates'))
        assert updates==epoch*152+step and state['loader']['next_batch']==updates
        assert state['loader']['draw_prefix_sha256']==prefix[updates]
        evaluations=state['evaluations'];training_log=state['loader']['training_log']
        assert len(training_log)==updates
    assert all(str(e) in evaluations for e in (0,5,10,15,20) if e<epoch), 'Missing historical eval: stop, never relabel new weights'

    def snapshot():
        return capture(model,optimizer,progress=dict(epoch=epoch,step=step,optimizer_updates=updates,boundary='full_update'),
            loader=dict(resume_strategy='fixed_old_C1_replay_by_global_update_no_prefetch',next_batch=updates,
                        seed=seed,draw_prefix_sha256=prefix[updates],draw_sha256=EXPECTED_DRAW,training_log=list(training_log)),
            evaluations=dict(evaluations))

    def files():
        p=ctx.work/'training-log.jsonl'
        p.write_text(''.join(json.dumps(r)+'\n' for r in training_log))
        c=ctx.work/'config-snapshot.json';c.write_text(json.dumps(ctx.config,indent=2)+'\n')
        sources=ctx.work/'experiment-source.zip'
        if not sources.exists():
            with zipfile.ZipFile(sources,'w',zipfile.ZIP_DEFLATED) as z:
                for q in sorted(Path(__file__).parent.glob('*.py')):z.write(q,q.name)
                z.write(root/'public-freeze.json','public-freeze.json')
        return {p.name:p,c.name:c,sources.name:sources}

    def evaluate(e):
        assert e==epoch and step==0
        model.eval();records=[]
        masks=ctx.work/f'predictions-{e:02d}.zip'
        with torch.inference_mode(),zipfile.ZipFile(masks,'w',zipfile.ZIP_DEFLATED) as archive:
            for i,row in enumerate(val.rows):
                ctx.check_budget()
                x,_,_,meta=val.item(i);_,mask=val.raw(i)
                pred=restore(model(x[None].to('cuda')),meta)[0,0].cpu().numpy()>0
                record={k:row[k] for k in ('id','center','group','kind')}
                record.update(score(pred,np.asarray(mask)>127));records.append(record)
                data=io.BytesIO();Image.fromarray(pred.astype('uint8')*255).save(data,format='PNG')
                archive.writestr(row['id']+'.png',data.getvalue())
        per=ctx.work/f'per-image-{e:02d}.jsonl';per.write_text(''.join(json.dumps(r)+'\n' for r in records))
        metric=summary(records,e);summ=ctx.work/f'summary-{e:02d}.json';summ.write_text(json.dumps(metric,indent=2)+'\n')
        artifacts={p.name:p for p in (masks,per,summ)}
        if e in (0,5,10,15,20):
            ck=ctx.work/f'epoch-{e:02d}.pt'
            torch.save(dict(model=model.state_dict(),arm=ctx.config['arm']+'_old_C1',epoch=e,updates=updates,
                            experiment_lock=ctx.envelope['experiment_lock'],draw_sha256=EXPECTED_DRAW),ck)
            receipt=ctx.work/f'epoch-{e:02d}-checkpoint.json'
            receipt.write_text(json.dumps(dict(file=ck.name,bytes=ck.stat().st_size,sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),epoch=e)))
            artifacts[receipt.name]=receipt
        evaluations[str(e)]=dict(artifacts=list(artifacts))
        ctx.boundary(snapshot(),dict(artifacts,**files()),force=True)
        print(json.dumps(metric),flush=True)

    if state is None:ctx.boundary(snapshot(),files(),force=True)
    if step==0 and epoch in (0,5,10,15,20) and str(epoch) not in evaluations:evaluate(epoch)
    while epoch<20:
        model.train()
        assert not model.encoder.training and all(not b.training for b in model.encoder.blocks)
        parameter_groups(model)
        for k in range(step,152):
            ctx.check_budget()
            torch.cuda.synchronize()
            started_step=time.monotonic()
            ids,seeds=replay[updates]
            x,y,valid=batch(train,ids,seeds);optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(x.to('cuda')),y.to('cuda'),valid.to('cuda'));assert torch.isfinite(loss)
            loss.backward()
            if updates==0:atomic_json(ctx.work/'gradient-audit.json',gradient_audit(model))
            optimizer.step();torch.cuda.synchronize()
            training_log.append(dict(epoch=epoch+1,step=k+1,update=updates+1,loss=float(loss.detach()),step_seconds=time.monotonic()-started_step,peak_vram_allocated=torch.cuda.max_memory_allocated(),peak_vram_reserved=torch.cuda.max_memory_reserved()))
            updates+=1;step=k+1
            # Capture only complete updates. Save at quarter-epoch boundaries.
            force=step%38==0
            ctx.boundary(snapshot(),files() if force else None,force=force)
        epoch+=1;step=0
        ctx.boundary(snapshot(),files(),force=True)
        if epoch in (0,5,10,15,20) and str(epoch) not in evaluations:evaluate(epoch)
    assert updates==3040 and '20' in evaluations
    final_encoder_sha256=encoder_digest(model)
    assert initial_encoder_sha256==final_encoder_sha256, 'Frozen encoder tensors changed'
    audit=ctx.work/'parameter-freeze-audit.json'
    audit.write_text(json.dumps(dict(encoder_sha256_initial=initial_encoder_sha256,encoder_sha256_final=final_encoder_sha256,all_original_dino_tensors_unchanged=True,trainable=sum(p.numel() for p in head),total=sum(p.numel() for p in model.parameters()),optimizer_allowed_only=True),indent=2))
    ctx.artifacts[audit.name]=audit
    ctx.boundary(snapshot(),files(),force=True)
