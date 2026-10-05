"""Extract the recorded 128 training-image patch features and compute linear CKA."""
import argparse, os, runpy, sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root',type=Path,required=True)
p.add_argument('--weights',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--sanity',action='store_true')
a=p.parse_args()
os.environ['POLYP_DATA_ROOT']=str(a.data_root.resolve())
os.environ['DINO_WEIGHTS']=str(a.weights.resolve())
os.environ['CKA_OUTPUT']=str(a.output.resolve())
sys.argv=[sys.argv[0]]+(['--sanity'] if a.sanity else [])
runpy.run_path(str(Path(__file__).resolve().parents[1]/'runtime/cka.py'),run_name='__main__')
