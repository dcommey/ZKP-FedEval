"""Compile supported shapes to measure constraints; does not time proving."""
import hashlib,json,re,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'circuits/committed_linear.circom').read_text();rows=[]
with tempfile.TemporaryDirectory(prefix='fedeval-shapes-') as temp:
 for b,d in [(4,16),(8,8),(8,16),(16,16)]:
  path=Path(temp)/f'shape_{b}_{d}.circom'
  path.write_text(source.replace('= CommittedLinear(8,16,10);',f'= CommittedLinear({b},{d},10);'))
  r=subprocess.run([str(ROOT/'node_modules/.bin/circom2'),str(path),'--O2','--r1cs','-l',str(ROOT/'node_modules'),'-o',temp],check=True,capture_output=True,text=True)
  clean=re.sub(r'\x1b\[[0-9;]*m','',r.stdout)
  count=int(re.search(r'non-linear constraints:\s*(\d+)',clean).group(1))
  rows.append({'batch_size':b,'dimensions':d,'classes':10,'nonlinear_constraints':count,
               'linear_constraints':int(re.search(r'^linear constraints:\s*(\d+)',clean,re.MULTILINE).group(1)),
               'proving_time_measured':False})
  print(b,d,count,flush=True)
assert next(r['nonlinear_constraints'] for r in rows if r['batch_size']==8 and r['dimensions']==16)==18282
out=ROOT/'results/circuit-shapes.json';out.write_text(json.dumps({'circuit_sha256':hashlib.sha256(source.encode()).hexdigest(),'optimization':'O2','shapes':rows},indent=2))
