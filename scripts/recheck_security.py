"""Rerun relation/request checks against current policy with an explicit test fixture."""
import json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from scripts.rerun_committed import node,loss_sum
OUT=ROOT/'results/utility-study';OUT.mkdir(parents=True,exist_ok=True)
x=np.arange(1,129,dtype=np.int64).reshape(8,16)
y=np.eye(10,dtype=np.int64)[:8]
w=(np.arange(160,dtype=np.int64).reshape(10,16)%4)-2;b=np.arange(10,dtype=np.int64)
q=loss_sum(x,y,w,b)
inp={'client_id':'1','salt':'123456789012345678901234567890','features':x.astype(str).tolist(),
 'labels':y.astype(str).tolist(),'weights':w.astype(str).tolist(),'bias':b.astype(str).tolist(),
 'threshold':str(q+1),'nonce':'20261001'}
inp['dataset_root']=node({'op':'enroll','items':[inp]},OUT/'functional-enroll.json')[0]
result=node({'op':'security','wasm':'zkp_setup/committed/committed_linear_js/committed_linear.wasm',
 'zkey':'zkp_setup/committed/committed_linear_final.zkey','vkey':'zkp_setup/committed/verification_key.json',
 'input':inp,'loss_sum':str(q),'verify_counts':[]},OUT/'security-checks.json')
assert len(result['checks'])==25 and all(c['passed'] for c in result['checks'])
result['fixture']='synthetic bounded labeled batch for functional/adversarial testing; not model-quality evidence'
(OUT/'security-checks.json').write_text(json.dumps(result,indent=2))
print('All 25 relation/request checks passed against current policy')
