"""
Server implementation for ZKP-based federated evaluation.
Manages client interactions, proof verification, and result aggregation.
"""

import torch
import time
import random
import string
import json # Needed for potential json parsing errors
import copy # For deepcopying model
import sys # For stderr
import torch.optim as optim # Import optimizer
import torch.nn.functional as F # Import loss function

# Import updated functions from zkp_utils
from zkp_utils import setup_zkp, verify_zkp, hash_model_weights, prepare_circuit_inputs

class Server:
    def __init__(self, model, clients, device, loss_threshold):
        self.model = model.to(device)
        self.clients = clients # List of Client objects (or None)
        self.device = device
        self.loss_threshold = loss_threshold
        self.zkp_setup_info = None
        self.total_comm_cost_bytes = 0
        self.total_server_verify_time = 0.0
        self.setup_done = False # Flag to track if setup completed successfully
        # Add optimizer (e.g., Adam)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.constraint_count = -1 # Initialize constraint count

    def train_model(self, train_loader, epochs=1):
        """Performs simple centralized training on the server."""
        if not train_loader:
            print("Server: No training data loader provided. Skipping training.", file=sys.stderr)
            return

        print(f"Server: Starting central training for {epochs} epoch(s)...", file=sys.stderr)
        self.model.train() # Set model to training mode
        start_time = time.time()
        for epoch in range(epochs):
            epoch_loss = 0.0
            samples_processed = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target) # Use standard CE loss for training
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * data.size(0)
                samples_processed += data.size(0)
                if batch_idx % 50 == 0: # Print progress occasionally
                    print(f"  Epoch {epoch+1}/{epochs} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}", file=sys.stderr)

            avg_epoch_loss = epoch_loss / samples_processed if samples_processed > 0 else 0
            print(f"Server: Epoch {epoch+1} completed. Average Training Loss: {avg_epoch_loss:.4f}", file=sys.stderr)

        elapsed_time = time.time() - start_time
        print(f"Server: Central training finished in {elapsed_time:.2f} seconds.", file=sys.stderr)
        self.model.eval() # Set model back to evaluation mode


    def setup_phase(self):
        """Performs the one-time ZKP setup using real tools. Sets setup_done flag."""
        # Only run setup once
        if self.setup_done:
            print("Server: ZKP setup already performed successfully.", file=sys.stderr)
            return True # Indicate setup is already done

        print("Server: Performing ZKP setup (compiling circuit, generating keys)...", file=sys.stderr)
        try:
            self.zkp_setup_info = setup_zkp()
            if not self.zkp_setup_info:
                 # setup_zkp should raise an error if it fails internally
                 raise RuntimeError("ZKP setup function returned None unexpectedly.")
            print("Server: ZKP setup complete.", file=sys.stderr)
            self.setup_done = True # Mark setup as completed successfully
            # Store constraint count
            self.constraint_count = self.zkp_setup_info.get("constraint_count", -1)
            return True
        except Exception as e:
            print(f"Server: ZKP setup FAILED: {e}", file=sys.stderr)
            print("Check prerequisites (Circom, snarkjs, Node.js, PTau file) and paths.", file=sys.stderr)
            self.zkp_setup_info = None # Ensure setup info is cleared on failure
            self.setup_done = False # Ensure flag reflects failure
            # Do not re-raise here, let the caller handle the False return
            return False

    def run_evaluation_round(self):
        """
        Runs one round of federated evaluation with real ZKP verification.
        Returns a dictionary of results or raises RuntimeError if setup failed.
        """
        if not self.setup_done or not self.zkp_setup_info:
             # This check ensures setup was attempted and succeeded
             print("Error: Server ZKP setup has not been completed successfully.", file=sys.stderr)
             raise RuntimeError("ZKP setup must be completed successfully before running evaluation round.")

        print("\n--- Starting Federated Evaluation Round ---", file=sys.stderr)
        self.total_comm_cost_bytes = 0
        self.total_server_verify_time = 0.0
        valid_proof_count = 0
        client_proof_gen_times = []
        client_comm_costs = [] # Store costs only for clients who submitted a proof
        server_verification_times = [] # Store times only for proofs that were attempted to verify

        # 1. Server Initialization for the round
        model_hash = hash_model_weights(self.model) # Get string hash
        round_nonce = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
        print(f"Server: Distributing Model (Hash: {model_hash[:8]}...), Threshold: {self.loss_threshold}, Nonce: {round_nonce}", file=sys.stderr)

        # Generate the *expected* public inputs list ONCE for this round using the same function
        try:
            # Only unpack the first element (the list)
            expected_public_inputs_list, _, _ = prepare_circuit_inputs(
                local_loss=0.0, # Dummy loss value, doesn't affect public inputs list structure
                threshold=self.loss_threshold,
                nonce=round_nonce,
                model_hash=model_hash
            )
            if expected_public_inputs_list is None:
                 raise ValueError("prepare_circuit_inputs returned None")
            print(f"Server: Expected Public Inputs (list format for verification): {expected_public_inputs_list}", file=sys.stderr)
        except Exception as e:
             print(f"Server: ERROR preparing expected public inputs: {e}", file=sys.stderr)
             raise RuntimeError("Failed to prepare expected public inputs for the round.") from e


        # 2. Distribute model to clients (ensure deep copy)
        for client in self.clients:
            if client:
                # Pass a deep copy to prevent clients from potentially modifying the server's model instance
                client.set_model(copy.deepcopy(self.model))

        # 3. Clients compute loss and generate proof (conceptually parallel)
        client_results = [] # Store results: ( (proof, public_list), size, time ) or (None, 0, time)
        active_clients_participated = 0
        for i, client in enumerate(self.clients):
             if client:
                 active_clients_participated += 1
                 print(f"\n--- Client {client.client_id} Processing ---", file=sys.stderr)
                 try:
                     # generate_threshold_proof returns ((proof, public_list), size, time) or (None, 0, time)
                     result = client.generate_threshold_proof(
                         self.zkp_setup_info,
                         self.loss_threshold,
                         round_nonce,
                         model_hash
                     )
                     client_results.append(result)
                     # Log time immediately
                     client_proof_gen_times.append(result[2]) # Store total time (eval + proof gen attempt)
                     # Log cost only if proof was generated
                     if result[0] is not None:
                          client_comm_costs.append(result[1])

                 except Exception as e:
                     print(f"Server: ERROR during client {client.client_id} execution: {e}", file=sys.stderr)
                     # Append a failure indicator, e.g., None, to maintain list length
                     client_results.append(None)
                     # Optionally record a time penalty or estimated time if needed
                     # client_proof_gen_times.append(some_default_or_error_time)
             else:
                  client_results.append(None) # Handle case where client object in list is None

        # 4. Server Verification and Aggregation
        print("\n--- Server Verification ---", file=sys.stderr)
        proofs_received_count = 0
        for i, result in enumerate(client_results):
            client_id = self.clients[i].client_id if self.clients[i] else f"Inactive_{i}"

            if result is None:
                 print(f"Server: No result received from Client {client_id} (likely error during client execution).", file=sys.stderr)
                 self.total_comm_cost_bytes += 10 # Assume small cost for timeout/error message
                 continue

            # Unpack result: proof_tuple is (proof_obj, received_public_list) or None
            proof_tuple, proof_size_bytes, _ = result # We already stored the time

            if proof_tuple: # Check if proof was generated (not None)
                proofs_received_count += 1
                proof_obj, received_public_inputs_list = proof_tuple
                self.total_comm_cost_bytes += proof_size_bytes # Add cost of received proof

                print(f"Server: Verifying proof from Client {client_id}...", file=sys.stderr)
                # print(f"Server: Received Public Inputs: {received_public_inputs_list}", file=sys.stderr) # Optional debug

                # **CRITICAL CHECK:** Compare received public inputs with server's expected list
                if received_public_inputs_list == expected_public_inputs_list:
                    print(f"Server: Public inputs match expected for Client {client_id}. Proceeding with verification.", file=sys.stderr)
                    try:
                        # verify_zkp returns (is_valid, verify_time) or raises Exception
                        is_valid, verify_time = verify_zkp(
                            self.zkp_setup_info,
                            received_public_inputs_list, # Use the list received from the client
                            proof_obj
                        )
                        server_verification_times.append(verify_time) # Record time for this verification attempt
                        self.total_server_verify_time += verify_time

                        if is_valid:
                            print(f"Server: Proof from Client {client_id} is VALID.", file=sys.stderr)
                            valid_proof_count += 1
                        else:
                            # verify_zkp returned False (e.g., snarkjs reported invalid proof)
                            print(f"Server: Proof from Client {client_id} is INVALID (verification failed).", file=sys.stderr)

                    except Exception as e:
                         # Catch errors from run_command within verify_zkp or other verification issues
                         print(f"Server: ERROR during verification for client {client_id}: {e}", file=sys.stderr)
                         # Do not record a time for failed verification attempts? Or record 0? Let's skip appending time here.
                         # server_verification_times.append(0) # Or skip
                else:
                    # Public inputs mismatch - proof is for a different statement!
                    print(f"Server: !! WARNING !! Public input mismatch for Client {client_id}.", file=sys.stderr)
                    print(f"  Expected: {expected_public_inputs_list}", file=sys.stderr)
                    print(f"  Received: {received_public_inputs_list}", file=sys.stderr)
                    print("  Proof considered INVALID (incorrect public statement). Verification skipped.", file=sys.stderr)
                    # Do not count this as valid, do not add verification time

            else:
                # Client evaluated loss >= threshold, or failed during proof gen after eval
                print(f"Server: No proof received from Client {client_id} (loss >= threshold or generation failed).", file=sys.stderr)
                # Assume small cost for "no proof" message or timeout
                self.total_comm_cost_bytes += 10 # ~cost of simple message

        # 5. Report Results and prepare return dictionary
        print("\n--- Evaluation Round Summary ---", file=sys.stderr)
        results = {
            "num_total_clients": len(self.clients),
            "num_active_clients": active_clients_participated,
            "proofs_received_count": proofs_received_count, # How many clients sent a proof
            "valid_proof_count": valid_proof_count, # How many proofs were successfully verified
            "loss_threshold": self.loss_threshold,
            "avg_client_proof_gen_time_s": sum(client_proof_gen_times) / len(client_proof_gen_times) if client_proof_gen_times else 0,
            "avg_server_verify_time_s": sum(server_verification_times) / len(server_verification_times) if server_verification_times else 0,
            "total_server_verify_time_s": self.total_server_verify_time,
            "avg_comm_cost_proof_kib": (sum(client_comm_costs) / len(client_comm_costs) / 1024.0) if client_comm_costs else 0,
            "total_comm_cost_upload_kib": self.total_comm_cost_bytes / 1024.0,
            "valid_proof_percentage": (valid_proof_count / active_clients_participated * 100) if active_clients_participated > 0 else 0,
            "model_hash": model_hash, # Include for reference
            "round_nonce": round_nonce, # Include for reference
            "constraint_count": self.constraint_count # Add constraint count
        }
        evaluation_times = [c.last_metrics['evaluation_s'] for c in self.clients
                            if c and 'evaluation_s' in c.last_metrics]
        proving_times = [c.last_metrics['witness_and_proving_s'] for c in self.clients
                        if c and c.last_metrics.get('witness_and_proving_s') is not None]
        results['avg_model_evaluation_time_s'] = sum(evaluation_times)/len(evaluation_times) if evaluation_times else None
        results['avg_witness_and_proving_time_s'] = sum(proving_times)/len(proving_times) if proving_times else None
        results['avg_client_processing_time_s'] = results['avg_client_proof_gen_time_s']
        results['guarantee'] = 'supplied-loss threshold only; not verified model evaluation'

        # Print summary to stderr
        print(f"Total Clients Configured: {results['num_total_clients']}", file=sys.stderr)
        print(f"Active Clients Participated: {results['num_active_clients']}", file=sys.stderr)
        print(f"Proofs Received by Server: {results['proofs_received_count']}", file=sys.stderr)
        print(f"Valid Proofs Verified: {results['valid_proof_count']} ({results['valid_proof_percentage']:.1f}% of active clients)", file=sys.stderr)
        print(f"Average Client Processing Time (Eval+ProofGen Attempt): {results['avg_client_proof_gen_time_s']:.3f}s", file=sys.stderr)
        print(f"Average Server Verification Time (per Attempted Proof): {results['avg_server_verify_time_s']:.4f}s", file=sys.stderr)
        print(f"Total Server Verification Time: {results['total_server_verify_time_s']:.4f}s", file=sys.stderr)
        print(f"Average Communication Cost (per Submitted Proof): {results['avg_comm_cost_proof_kib']:.2f} KiB", file=sys.stderr)
        print(f"Total Communication Cost (Upload): {results['total_comm_cost_upload_kib']:.2f} KiB", file=sys.stderr)
        print(f"ZKP Circuit Constraints: {results['constraint_count']}", file=sys.stderr) # Print constraint count


        # Return the collected results dictionary (will be printed as JSON by main.py)
        return results
