import torch
import argparse
import numpy as np
import sys
import json

from models import get_model
from data_loader import get_data_loaders
from client import Client
from server import Server

def main(args):
    run_params = vars(args)
    final_output = {"params": run_params}

    try:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

        print(f"Loading {args.dataset} dataset...", file=sys.stderr)
        try:
            server_train_loader, client_eval_loaders = get_data_loaders(
                dataset_name=args.dataset,
                num_clients=args.num_clients,
                batch_size=args.batch_size,
                data_percentage=args.data_percentage,
                non_iid=args.non_iid,
                server_train_fraction=0.1
            )
            print("Data loading complete.", file=sys.stderr)
        except Exception as e:
            print(f"CRITICAL ERROR during data loading: {e}", file=sys.stderr)
            error_output = {"error": "Data loading failed", "details": str(e), "params": vars(args)}
            print(json.dumps(error_output))
            sys.exit(1)

        try:
            model = get_model(args.dataset)
        except Exception as e:
            print(f"CRITICAL ERROR during model initialization: {e}", file=sys.stderr)
            error_output = {"error": "Model initialization failed", "details": str(e), "params": vars(args)}
            print(json.dumps(error_output))
            sys.exit(1)

        try:
            clients = [Client(client_id=i, data_loader=loader, device=device)
                       for i, loader in enumerate(client_eval_loaders)]
        except Exception as e:
            print(f"CRITICAL ERROR during client initialization: {e}", file=sys.stderr)
            error_output = {"error": "Client initialization failed", "details": str(e), "params": vars(args)}
            print(json.dumps(error_output))
            sys.exit(1)

        print("Initializing server...", file=sys.stderr)
        try:
            server = Server(model, clients, device, args.loss_threshold)
        except Exception as e:
            print(f"CRITICAL ERROR during server initialization: {e}", file=sys.stderr)
            final_output["error"] = "Server initialization failed"
            final_output["error_details"] = str(e)
            print(json.dumps(final_output))
            sys.exit(1)

        try:
            server.train_model(server_train_loader, epochs=1)
        except Exception as e:
            print(f"CRITICAL ERROR during server model training: {e}", file=sys.stderr)
            final_output["error"] = "Server training failed"
            final_output["error_details"] = str(e)
            print(json.dumps(final_output))
            sys.exit(1)

        print("Attempting ZKP setup via server...", file=sys.stderr)
        try:
            server.setup_phase()
        except Exception as e:
            print(f"CRITICAL ERROR during ZKP setup call (caught in main): {e}", file=sys.stderr)
            final_output["error"] = "ZKP setup failed during call"
            final_output["error_details"] = str(e)
            print(json.dumps(final_output))
            sys.exit(1)

        if not server.setup_done:
            print("Exiting because server.setup_done is False after setup_phase call.", file=sys.stderr)
            final_output["error"] = "ZKP setup failed"
            final_output["error_details"] = "Server setup_done flag is False. Check stderr logs for setup errors."
            print(json.dumps(final_output))
            sys.exit(1)

        round_results = None
        print("Running federated evaluation round...", file=sys.stderr)
        try:
            round_results = server.run_evaluation_round()
            print("Federated evaluation round finished.", file=sys.stderr)
        except Exception as e:
            print(f"ERROR during evaluation round (caught in main): {e}", file=sys.stderr)
            final_output["error"] = "Evaluation round failed"
            final_output["error_details"] = str(e)

        if round_results:
            final_output.update(round_results)

        try:
            print(json.dumps(final_output))
        except Exception as e:
            print(f"ERROR: Failed to serialize final results to JSON: {e}", file=sys.stderr)
            fallback_error = {"error": "JSON serialization failed", "params": run_params, "details": str(e)}
            print(json.dumps(fallback_error))
            sys.exit(1)

    except Exception as e:
        print(f"CRITICAL UNHANDLED ERROR in main: {e}", file=sys.stderr)
        final_output["error"] = "Unhandled exception in main"
        final_output["error_details"] = str(e)
        if "params" not in final_output: final_output["params"] = run_params
        print(json.dumps(final_output))
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZKP Federated Evaluation Simulation")

    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'har'],
                        help='Dataset to use (mnist or har)')
    parser.add_argument('--data-percentage', type=float, default=0.1, dest='data_percentage',
                        help='Percentage of the dataset to use (0.0 to 1.0)')
    parser.add_argument('--num-clients', type=int, default=5, dest='num_clients',
                        help='Number of clients')
    parser.add_argument('--batch-size', type=int, default=32, dest='batch_size',
                        help='Batch size for local evaluation')
    parser.add_argument('--non-iid', action='store_true', dest='non_iid',
                        help='Simulate non-IID data distribution')
    parser.add_argument('--loss-threshold', type=float, default=1.0, dest='loss_threshold',
                        help='Loss threshold for ZKP proof generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-gpu', action='store_true', dest='use_gpu',
                        help='Use GPU if available')

    args = parser.parse_args()

    main(args)