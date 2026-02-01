"""
Experiment runner for ZKP Federated Evaluation.
Executes evaluation experiments with different configurations and saves results.
"""

import os
import json
import time
import argparse
import torch
import random
import numpy as np
from datetime import datetime
import sys
import copy
from server import Server
from client import Client
from data_loader import get_data_loaders
from models import get_model
from zkp_utils import SCALE_FACTOR, setup_zkp

def set_random_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def set_zkp_precision(precision):
    """Sets the ZKP_PRECISION environment variable."""
    try:
        precision_int = int(precision)
        if precision_int < 1:
            raise ValueError("Precision must be at least 1.")
        os.environ['ZKP_PRECISION'] = str(precision_int)
        print(f"Set ZKP_PRECISION environment variable to: {precision_int}", file=sys.stderr)
    except ValueError as e:
        print(f"Error setting precision: {e}. Using default.", file=sys.stderr)
        if 'ZKP_PRECISION' in os.environ:
            del os.environ['ZKP_PRECISION'] # Use default from zkp_utils

def main():
    """Run experiments with multiple configurations and save results."""
    parser = argparse.ArgumentParser(description="Run ZKP Federated Evaluation experiments")
    parser.add_argument('--dataset', type=str, nargs='+', default=['mnist'], choices=['mnist', 'har'],
                        help='Dataset(s) to use (mnist or har)')
    parser.add_argument('--num-clients', type=int, nargs='+', default=[10],
                        help='List of client numbers for standard runs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training and evaluation')
    parser.add_argument('--loss-threshold', type=float, nargs='+', default=[1.0],
                        help='List of loss thresholds for proof generation')
    parser.add_argument('--data-percentage', type=float, default=0.1,
                        help='Percentage of dataset to use for evaluation clients (0.0 to 1.0)')
    parser.add_argument('--non-iid', action='store_true',
                        help='Use non-IID data distribution')
    parser.add_argument('--train-epochs', type=int, default=1,
                        help='Number of initial server training epochs')
    parser.add_argument('--num-seeds', type=int, default=1,
                        help='Number of random seeds to run per configuration')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--precision', type=int, nargs='+', default=[6], help='List of fixed-point precision (decimal places) for ZKP.')
    parser.add_argument('--client_counts', type=int, nargs='+', default=None, help='List of client numbers for scalability testing.')
    parser.add_argument('--scalability_mode', action='store_true', help='Run in scalability mode (vary clients, fixed dataset/threshold).')
    parser.add_argument('--precision_mode', action='store_true', help='Run in precision mode (vary precision, fixed clients/threshold).')

    args = parser.parse_args()

    # --- Input Validation ---
    if args.scalability_mode and args.precision_mode:
        print("Error: Cannot run in both scalability and precision mode simultaneously.", file=sys.stderr)
        sys.exit(1)
    if args.scalability_mode and args.client_counts is None:
        print("Error: --client_counts must be provided for scalability mode.", file=sys.stderr)
        sys.exit(1)
    if args.precision_mode and (len(args.num_clients) > 1 or len(args.loss_threshold) > 1):
         print("Warning: In precision mode, using first specified client count and threshold.", file=sys.stderr)
         args.num_clients = [args.num_clients[0]]
         args.loss_threshold = [args.loss_threshold[0]]

    # --- Experiment Setup ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_filename = os.path.join(results_dir, f"experiment_results_{timestamp}.json")
    all_results = []

    # Determine iteration loops based on mode
    datasets_to_run = args.dataset
    client_counts_to_run = args.num_clients if not args.scalability_mode else args.client_counts
    thresholds_to_run = args.loss_threshold
    precisions_to_run = args.precision if args.precision_mode else [args.precision[0]]
    seeds_to_run = range(args.seed, args.seed + args.num_seeds)
    non_iid_flags = [True] if args.non_iid else [False]

    # Override parameters for specific modes
    if args.scalability_mode:
        datasets_to_run = ['mnist']
        thresholds_to_run = [1.0]
        precisions_to_run = [6]
        non_iid_flags = [False]
        print("--- Running in Scalability Mode ---", file=sys.stderr)
        print(f"Datasets: {datasets_to_run}, Thresholds: {thresholds_to_run}, Precisions: {precisions_to_run}, Clients: {client_counts_to_run}", file=sys.stderr)

    if args.precision_mode:
        client_counts_to_run = [args.num_clients[0]]
        thresholds_to_run = [args.loss_threshold[0]]
        non_iid_flags = [False]
        print("--- Running in Precision Mode ---", file=sys.stderr)
        print(f"Datasets: {datasets_to_run}, Thresholds: {thresholds_to_run}, Precisions: {precisions_to_run}, Clients: {client_counts_to_run}", file=sys.stderr)

    # --- Main Experiment Loop ---
    total_runs = 0
    for precision in precisions_to_run:
        for dataset in datasets_to_run:
            for num_clients in client_counts_to_run:
                 for threshold in thresholds_to_run:
                      for non_iid in non_iid_flags:
                           total_runs += len(seeds_to_run)

    current_run = 0

    for precision in precisions_to_run:
        print(f"\n===== Setting ZKP Precision to {precision} =====\n", file=sys.stderr)
        set_zkp_precision(precision)

        initial_zkp_setup_info = None
        try:
            print("Performing ZKP setup for precision level...", file=sys.stderr)
            initial_zkp_setup_info = setup_zkp()
            if not initial_zkp_setup_info:
                 raise RuntimeError("setup_zkp returned None")
            print("Initial ZKP setup successful.", file=sys.stderr)
        except Exception as e:
            print(f"FATAL: Initial ZKP setup failed for precision {precision}. Skipping runs for this precision. Error: {e}", file=sys.stderr)
            runs_to_skip = 0
            for dataset in datasets_to_run:
                 for num_clients in client_counts_to_run:
                      for threshold in thresholds_to_run:
                           for non_iid in non_iid_flags:
                                runs_to_skip += len(seeds_to_run)
            current_run += runs_to_skip
            continue

        for dataset in datasets_to_run:
            for num_clients in client_counts_to_run:
                for threshold in thresholds_to_run:
                    for non_iid in non_iid_flags:
                        for seed in seeds_to_run:
                            current_run += 1
                            print(f"\n--- Running Experiment ({current_run}/{total_runs}) ---", file=sys.stderr)
                            run_params = {
                                "dataset": dataset,
                                "data_percentage": args.data_percentage,
                                "num_clients": num_clients,
                                "batch_size": args.batch_size,
                                "non_iid": non_iid,
                                "loss_threshold": threshold,
                                "seed": seed,
                                "use_gpu": args.use_gpu,
                                "precision": precision,
                                "train_epochs": args.train_epochs
                            }
                            print(f"Parameters: {run_params}", file=sys.stderr)

                            try:
                                set_random_seed(seed)
                                device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
                                print(f"Using device: {device}", file=sys.stderr)

                                print(f"Loading data for {dataset}...", file=sys.stderr)
                                server_train_loader, client_eval_loaders = get_data_loaders(
                                    dataset_name=dataset,
                                    num_clients=num_clients,
                                    batch_size=args.batch_size,
                                    data_percentage=args.data_percentage,
                                    non_iid=non_iid,
                                    server_train_fraction=0.1
                                )
                                print("Data loading complete.", file=sys.stderr)

                                model = get_model(dataset)
                                model.to(device)

                                clients = []
                                for i, loader in enumerate(client_eval_loaders):
                                     if loader is not None:
                                          clients.append(Client(client_id=i, data_loader=loader, device=device))
                                     else:
                                          print(f"Warning: Client {i} received no data loader, skipping client creation.", file=sys.stderr)

                                active_clients = [c for c in clients if c is not None]
                                if len(active_clients) == 0:
                                     raise ValueError("No active clients could be created (likely due to data distribution issues).")

                                server = Server(model, active_clients, device, threshold)
                                server.zkp_setup_info = initial_zkp_setup_info
                                server.setup_done = True
                                server.constraint_count = initial_zkp_setup_info.get("constraint_count", -1)

                                if args.train_epochs > 0 and server_train_loader:
                                     print(f"Starting initial server training for {args.train_epochs} epochs...", file=sys.stderr)
                                     server.train_model(server_train_loader, epochs=args.train_epochs)
                                     print("Initial server training complete.", file=sys.stderr)
                                elif args.train_epochs > 0:
                                     print("Warning: Server training requested but no server_train_loader available.", file=sys.stderr)

                                round_results = server.run_evaluation_round()

                                experiment_result = {
                                    "params": run_params,
                                    **round_results
                                }
                                all_results.append(experiment_result)

                                try:
                                    with open(results_filename, 'w') as f:
                                        json.dump(all_results, f, indent=2)
                                except Exception as json_e:
                                     print(f"Error saving results to JSON: {json_e}", file=sys.stderr)

                            except Exception as e:
                                print(f"Error during experiment run {current_run}: {e}", file=sys.stderr)
                                import traceback
                                traceback.print_exc(file=sys.stderr)
                                error_result = {
                                    "params": run_params,
                                    "error": str(e)
                                }
                                all_results.append(error_result)
                                try:
                                    with open(results_filename, 'w') as f:
                                        json.dump(all_results, f, indent=2)
                                except Exception as json_e:
                                     print(f"Error saving error results to JSON: {json_e}", file=sys.stderr)

    print(f"\n--- Experiment Complete ---", file=sys.stderr)
    print(f"Results saved to {results_filename}", file=sys.stderr)

if __name__ == "__main__":
    main()