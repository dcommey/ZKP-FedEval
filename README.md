# ZKP-FedEval

ZKP-FedEval is a research prototype for verifiable evaluation in federated learning.
A client proves that a public model passes or fails a loss limit on its private data.
The client does not show the data to the coordinator.

## What the proof shows

The main circuit is `circuits/committed_linear.circom`. It uses Groth16 on the BN254 curve.

Each proof shows these facts:

- The client used the full batch that was registered before the request.
  The batch has 8 records with 16 features and a one-hot label.
- The circuit calculated the squared loss of the public linear model on each record.
- The output bit is 1 if the loss is less than the limit, and 0 if not.
- The proof is bound to the client ID, the model, the limit, and the round number.

A proof with output 0 is a proved failure.
A missing proof is not a failure. It is a missing result.

## What the proof does not show

- The proof does not show that the records are real.
  A provisioning authority must select and register the records.
- The provisioning authority knows the records.
  It must not share them with the coordinator.
- Feature extraction (PCA) and model training occur outside the circuit.
- The circuit does not prove CNN or MLP models. It proves a linear model only.
- The trusted setup is a single-party research setup.
  Do not use it in production.

## Query limit and release rule

`scripts/protocol_policy.cjs` contains two policies.

- **Client policy:** Each registered batch gives one evaluation only.
  This limit applies to all models and all rounds.
  The limit stops a coordinator from finding the loss with many queries.
- **Coordinator policy:** The coordinator accepts each proof one time only.
  It counts proved passes, proved failures, and missing results.
  It approves a release only when the number of proved passes is sufficient.

## Requirements

- Python 3.10 or later
- Node.js 18 or later

The npm packages include `circom2` and `snarkjs`. You do not have to install them separately.
The scripts download MNIST and UCI HAR automatically into `data/`.

## Install

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
npm ci
```

## Run the experiments

Run the commands in this sequence from the repository root.

1. Make the research keys. This step takes some minutes.

   ```bash
   .venv/bin/python scripts/setup_committed.py
   ```

   ```bash
   .venv/bin/python scripts/setup_threshold.py
   ```

2. Measure the supplied-loss baseline circuit.

   ```bash
   node scripts/benchmark_threshold.cjs
   ```

3. Make the 540 main proofs. This step takes approximately 30 minutes.

   ```bash
   .venv/bin/python scripts/rerun_committed.py
   ```

4. Do the model-quality study. This step makes 18 more proofs.

   ```bash
   .venv/bin/python scripts/utility_study.py
   ```

5. Do the policy and security checks.

   ```bash
   node scripts/check_policy.cjs
   ```

   ```bash
   node scripts/check_release.cjs
   ```

   ```bash
   .venv/bin/python scripts/recheck_security.py
   ```

6. Count the constraints for other circuit sizes.

   ```bash
   .venv/bin/python scripts/circuit_shapes.py
   ```

7. Make the summary and the figures.

   ```bash
   .venv/bin/python scripts/build_summary.py
   ```

   ```bash
   .venv/bin/python scripts/render_figures.py
   ```

8. Run the unit tests.

   ```bash
   .venv/bin/python -m pytest tests
   ```

New keys and new salts give new commitments. Timing values change between computers and runs.

## Results

The `results/` folder contains our measured results.

| File | Contents |
|---|---|
| `results/committed-rerun/proof_measurements.json` | Time and size of each of the 540 proofs |
| `results/committed-rerun/local_outcomes.json` | Loss and output bit of each batch |
| `results/committed-rerun/model_quality.json` | Accuracy and loss of each model |
| `results/utility-study/` | Model-quality study and its 18 proofs |
| `results/policy-checks.json` | Results of the 17 policy checks |
| `results/release-policy.json` | Results of the 12 release-rule checks |
| `results/summary.json` | Summary of all results and checks |
| `zkp_setup/*/verification_key.json` | Verification keys |

The results do not contain private salts, witnesses, or proving keys.

## Earlier pipeline

The files `main.py`, `experiment_runner.py`, `client.py`, and `server.py` are the earlier pipeline.
This pipeline uses `circuits/loss_threshold.circom`.
That circuit only compares a loss value that the client supplies with the limit.
It does not calculate the loss from the model and the data.
We keep it as a baseline for comparison.

## Citation

The arXiv paper below describes an earlier version of this code.
The current circuit and its trust model are different.

```bibtex
@misc{commey2025zkpfedeval,
  title={ZKP-FedEval: Verifiable and Privacy-Preserving Federated Evaluation using Zero-Knowledge Proofs},
  author={Daniel Commey and Benjamin Appiah and Griffith S. Klogo and Garth V. Crosby},
  year={2025},
  eprint={2507.11649},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2507.11649}
}
```

## Contact

Daniel Commey, California State University, Long Beach: daniel.commey@csulb.edu

## License

This project uses the MIT License. See `LICENSE`.
