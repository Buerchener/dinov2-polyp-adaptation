"""Offline CPU tests for LoRA gradients, invariants and metric semantics."""
import io,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))

def main():
    import torch
    from models import DinoStudent,counts
    from lora_model import LoRAStudent,parameter_groups,encoder_digest,gradient_audit,is_lora
    from micro_metrics import aggregate
    torch.set_num_threads(2)
    torch.manual_seed(20261002);base=DinoStudent(False)
    torch.manual_seed(20261002);model=LoRAStudent(False)
    assert counts(model)['trainable']==541377
    assert sum(p.numel() for n,p in model.named_parameters() if is_lora(n))==73728
    model.train();base.train()
    assert not model.encoder.training
    original=encoder_digest(model)
    x=torch.rand(1,3,56,56)
    with torch.no_grad():assert torch.equal(base(x),model(x)), 'Zero-B initialization must preserve baseline'
    opt=torch.optim.AdamW(parameter_groups(model),weight_decay=1e-4)
    target=torch.rand(1,1,56,56)
    logits=model(x);torch.nn.functional.binary_cross_entropy_with_logits(logits,target).backward()
    gradient_audit(model)
    assert any(p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.endswith('.lora_B'))
    opt.step();opt.zero_grad(set_to_none=True)
    torch.nn.functional.binary_cross_entropy_with_logits(model(x),target).backward();gradient_audit(model)
    assert any(p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.endswith('.lora_A'))
    assert encoder_digest(model)==original
    buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
    replica=LoRAStudent(False);replica.load_state_dict(torch.load(buf,weights_only=True),strict=True)
    model.eval();replica.eval()
    with torch.no_grad():
        assert torch.equal(model(x),replica(x))
        assert model(torch.rand(1,3,448,448)).shape==(1,1,448,448)
    rows=[dict(id='large',reference_positive=True,intersection=90,prediction_pixels=100,gt_pixels=100,image_pixels=200),dict(id='miss',reference_positive=True,intersection=0,prediction_pixels=0,gt_pixels=1,image_pixels=200),dict(id='normal',reference_positive=False,intersection=0,prediction_pixels=200,gt_pixels=0,image_pixels=200)]
    m=aggregate(rows)
    assert m['positive_micro_dice']==180/201 and m['positive_macro_dice']==.45
    assert aggregate(rows[:2])['positive_micro_dice']==m['positive_micro_dice']
    assert m['normal_any_pixel_fp']==1 and m['normal_mean_foreground']==1 and m['empty_positive_predictions']==1
    for bad in [rows+rows[:1],[dict(rows[0],intersection=-1)],[dict(rows[0],reference_positive=False)]]:
        try:aggregate(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid counts accepted')
    m=aggregate([json.loads(s) for s in (ROOT/'results/final/lora_final.jsonl').read_text().splitlines()])
    assert (m['TP'],m['FP'],m['FN'])==(25851756,1782932,4481457)
    print(json.dumps(dict(status='PASS',device='cpu',torch=torch.__version__,checks=['zero-B baseline equivalence','541377/73728 parameter counts','LoRA A/B gradients over two backward passes','original encoder unchanged after optimizer step','strict checkpoint roundtrip','448 output geometry','micro/macro/normal separation','invalid count rejection','published final confusion counts'])))
if __name__=='__main__':main()
