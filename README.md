# ZKP-FedEval

This repository contains the reference implementation for our ZKP-based federated evaluation experiments.
It includes data loading, centralized server training (for an initial model), client-side evaluation, Groth16 proof generation/verification via Circom + snarkjs, and analysis scripts.

## What the current ZKP proves
The included Circom circuit proves a **threshold statement** about a client-provided loss value:

- The prover knows a private `calculated_loss` such that `calculated_loss < threshold`.
- The public inputs include `(model_hash, threshold, nonce)`.

Important: the current circuit is a lightweight benchmark circuit and **does not** compute the model forward pass or loss inside the circuit. It therefore does not, by itself, guarantee that the loss corresponds to the provided model and dataset.
## Features
- **Experiment runner**: Runs sweeps over client count, thresholds, and fixed-point precision.
- **Dataset support**: MNIST and UCI HAR.
- **ZKP integration**: Circom + snarkjs (Groth16) for proof generation/verification.
- **Analysis tooling**: Plots and summary tables for runtime and communication metrics.

## Setup
### Python
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### ZKP toolchain
Ensure these are installed and on `PATH`:
- `node` (Node.js)
- `snarkjs`
- `circom`

This repo expects a Powers of Tau file at `zkp_artifacts/pot12_final.ptau`.

## Usage
Single run:
```bash
python3 main.py --dataset mnist --num-clients 10 --loss-threshold 1.0
```

Experiment sweep:
```bash
python3 experiment_runner.py --dataset mnist --num-clients 10 20 50 --loss-threshold 0.5 1.0 --num-seeds 3
```

## Results
Results are saved under `results/`. Use `analyze_results.py` to generate plots/tables into `analysis_output/`.

## License
This project is licensed under the MIT License.