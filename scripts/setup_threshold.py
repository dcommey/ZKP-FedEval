"""Set up the supplied-loss comparison using the research Powers of Tau."""
import os
from pathlib import Path
import secrets
import subprocess

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
BIN = ROOT / 'node_modules' / '.bin'
OUT = ROOT / 'zkp_setup' / 'threshold'
PTAU = ROOT / 'zkp_artifacts' / 'pot15_final.ptau'
if not PTAU.exists():
    raise SystemExit('Run scripts/setup_committed.py first to create research setup parameters.')
OUT.mkdir(parents=True, exist_ok=True)

def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)

run(BIN / 'circom2', 'circuits/loss_threshold.circom', '--O2', '--r1cs', '--wasm',
    '--sym', '-l', 'node_modules', '-o', OUT)
run(BIN / 'snarkjs', 'groth16', 'setup', OUT / 'loss_threshold.r1cs', PTAU, OUT / 'initial.zkey')
run(BIN / 'snarkjs', 'zkey', 'contribute', OUT / 'initial.zkey', OUT / 'final.zkey',
    '--name=local research baseline', '-e=' + secrets.token_hex(64))
run(BIN / 'snarkjs', 'zkey', 'verify', OUT / 'loss_threshold.r1cs', PTAU, OUT / 'final.zkey')
run(BIN / 'snarkjs', 'zkey', 'export', 'verificationkey', OUT / 'final.zkey', OUT / 'verification_key.json')
