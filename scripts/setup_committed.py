"""Single-party research setup. NOT a production Groth16 ceremony."""
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import os

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
BIN=ROOT/'node_modules'/'.bin'
ART=ROOT/'zkp_artifacts'
SETUP=ROOT/'zkp_setup'/'committed'
ART.mkdir(exist_ok=True);SETUP.mkdir(parents=True,exist_ok=True)
def run(*args): subprocess.run([str(x) for x in args],check=True)
run(BIN/'circom2','circuits/committed_linear.circom','--O2','--r1cs','--wasm','--sym','-l','node_modules','-o',SETUP)
ptau=ART/'pot15_final.ptau'
if not ptau.exists():
    run(BIN/'snarkjs','powersoftau','new','bn128',15,ART/'pot15_0000.ptau')
    run(BIN/'snarkjs','powersoftau','contribute',ART/'pot15_0000.ptau',ART/'pot15_0001.ptau',
        '--name=local research setup','-e='+secrets.token_hex(64))
    run(BIN/'snarkjs','powersoftau','prepare','phase2',ART/'pot15_0001.ptau',ptau)
run(BIN/'snarkjs','powersoftau','verify',ptau)
run(BIN/'snarkjs','groth16','setup',SETUP/'committed_linear.r1cs',ptau,SETUP/'initial.zkey')
run(BIN/'snarkjs','zkey','contribute',SETUP/'initial.zkey',SETUP/'committed_linear_final.zkey',
    '--name=local research phase2','-e='+secrets.token_hex(64))
run(BIN/'snarkjs','zkey','verify',SETUP/'committed_linear.r1cs',ptau,SETUP/'committed_linear_final.zkey')
run(BIN/'snarkjs','zkey','export','verificationkey',SETUP/'committed_linear_final.zkey',SETUP/'verification_key.json')
manifest={'ceremony':'single-party research-only; not a production multiparty setup',
          'optimization':'O2','nonlinear_constraints':18282,'public_inputs':174,'public_outputs':2}
for p in [Path('circuits/committed_linear.circom'),SETUP/'committed_linear.r1cs',ptau,SETUP/'committed_linear_final.zkey']:
    manifest[str(p.relative_to(ROOT) if p.is_absolute() else p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(SETUP/'setup_manifest.json').write_text(json.dumps(manifest,indent=2))
