"""Portable context around the recorded Phase 4 training loop. No cloud control."""
import argparse, hashlib, json, os, sys, time, signal
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'runtime'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['Baseline_Fusion','DetailSkip_Fusion','C_LoRA_LR5e5'],default='Baseline_Fusion')
    p.add_argument('--max-seconds',type=int,required=True)
    a=p.parse_args()
    if a.max_seconds<=0:p.error('--max-seconds must be positive')
    if not a.weights.is_file():p.error('Pretrained weights file does not exist')
    expected_weight=json.loads((REPO/'configs/cka/protocol.json').read_text())['weight_sha256']
    if hashlib.sha256(a.weights.read_bytes()).hexdigest()!=expected_weight:p.error('Pretrained weight checksum differs from the historical protocol')
    os.environ['POLYP_DATA_ROOT']=str(a.data_root.resolve())
    os.environ['DINO_WEIGHTS']=str(a.weights.resolve())
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch, timm, numpy, PIL
    if a.arm == 'C_LoRA_LR5e5':
        from lora_train_entry import run
    else:
        from train_entry import run
    from full_state import Store
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'environment.json').write_text(json.dumps(dict(torch=torch.__version__,timm=timm.__version__,numpy=numpy.__version__,pillow=PIL.__version__,cuda=torch.version.cuda),indent=2))
    class Context:
        def __init__(self):
            self.work=a.output
            self.config=json.loads((REPO/('configs/lora_final.json' if a.arm == 'C_LoRA_LR5e5' else 'configs/frozen.json')).read_text())
            self.config.update(arm=a.arm,target_epoch=20,required_evaluations=[0,5,10,15,20])
            lock=hashlib.sha256((REPO/'runtime/public-freeze.json').read_bytes()+json.dumps(self.config,sort_keys=True).encode()).hexdigest()
            self.envelope=dict(experiment_lock=lock)
            self.store=Store(self.work/'checkpoints',lock)
            self.artifacts={};self.deadline=time.monotonic()+a.max_seconds;self.stop=False
            signal.signal(signal.SIGTERM,self.stop_requested)
            signal.signal(signal.SIGINT,self.stop_requested)
        def stop_requested(self,*unused):self.stop=True
        def load(self):return None
        def check_budget(self):
            if self.stop or time.monotonic()>=self.deadline:raise TimeoutError('Run stopped; inspect saved full-state generations before a separate recovery')
        def boundary(self,state,artifacts=None,force=False):
            if artifacts:self.artifacts.update(artifacts)
            if force:self.store.commit(state,self.artifacts)
            self.check_budget()
    run(Context())
if __name__=='__main__':main()
