import torch
import torch.nn.functional as F
import time
import copy
import sys # For stderr

# Import the updated functions
from zkp_utils import generate_zkp, prepare_circuit_inputs

class Client:
    def __init__(self, client_id, data_loader, device):
        self.client_id = client_id
        self.data_loader = data_loader
        self.device = device
        self.model = None
        self.last_metrics = {}
        self.loss_func = F.cross_entropy  # Using sum reduction for precise average calculation

    def set_model(self, global_model):
        """Sets the client's model from the global model."""
        self.model = copy.deepcopy(global_model).to(self.device)
        self.model.eval()

    def evaluate_locally(self):
        """Evaluates the model on local data and returns the average loss."""
        if not self.model:
            print(f"Client {self.client_id}: Model not set.", file=sys.stderr)
            return float('inf')
        if not self.data_loader:
            print(f"Client {self.client_id}: No data loader available.", file=sys.stderr)
            return float('inf')

        total_loss = 0.0
        total_samples = 0
        self.model.eval()

        with torch.no_grad():
            try:
                for data, target in self.data_loader:
                    if data.size(0) == 0:
                        continue
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.loss_func(output, target, reduction='sum')
                    total_loss += loss.item()
                    total_samples += data.size(0)
            except Exception as e:
                print(f"Client {self.client_id}: Error during local evaluation: {e}", file=sys.stderr)
                return float('inf')

        if total_samples == 0:
            print(f"Client {self.client_id}: No samples found in data loader for evaluation.", file=sys.stderr)
            return float('inf')

        average_loss = total_loss / total_samples
        print(f"Client {self.client_id}: Local evaluation - Avg Loss: {average_loss:.6f} ({total_samples} samples)", file=sys.stderr)
        return average_loss

    def generate_threshold_proof(self, zkp_setup_info, threshold, nonce, model_hash):
        """
        Evaluates loss and generates ZKP proof if loss < threshold using real ZKP tools.
        Returns: Tuple ( (proof_obj, public_inputs_list), proof_size_bytes, total_time )
                 Returns (None, 0, total_time) if loss >= threshold or proof generation fails.
        """
        start_time = time.time()
        proof_result_tuple = None
        proof_size_bytes = 0

        # 1. Perform local evaluation to get the single average loss value
        local_loss = self.evaluate_locally()
        self.last_metrics = {'evaluation_s':time.time()-start_time,
                             'witness_and_proving_s':None,'status':'no_proof'}

        # Handle potential infinite loss from evaluation issues
        if local_loss == float('inf'):
             print(f"Client {self.client_id}: Local evaluation failed or yielded no result. Cannot generate proof.", file=sys.stderr)
             total_time = time.time() - start_time
             return None, 0, total_time

        if local_loss >= threshold:
            print(f"Client {self.client_id}: Loss {local_loss:.6f} >= threshold {threshold}. No proof generated.", file=sys.stderr)
            total_time = time.time() - start_time
            return None, 0, total_time # No proof, 0 bytes, elapsed time

        print(f"Client {self.client_id}: Loss {local_loss:.6f} < threshold {threshold}. Preparing ZKP inputs...", file=sys.stderr)

        # 2. Prepare inputs for the ZKP circuit
        try:
            # Correctly unpack the three return values
            public_inputs_list, public_inputs_dict, private_inputs_dict = prepare_circuit_inputs(
                local_loss=local_loss,
                threshold=threshold,
                nonce=nonce,
                model_hash=model_hash
            )
            # Check if preparation failed
            if public_inputs_list is None:
                 raise ValueError("prepare_circuit_inputs returned None")

            print(f"Client {self.client_id}: Inputs prepared. Generating proof...", file=sys.stderr)
        except Exception as e:
            print(f"Client {self.client_id}: Error preparing circuit inputs: {e}", file=sys.stderr)
            total_time = time.time() - start_time
            return None, 0, total_time

        # 3. Call ZKP generation utility
        try:
            # Pass the correct dictionaries to generate_zkp
            proof_result_tuple, proof_size_bytes, prove_time = generate_zkp(
                zkp_setup_info,
                public_inputs_dict, # Pass the public dictionary
                private_inputs_dict # Pass the private dictionary
            )
            self.last_metrics['witness_and_proving_s'] = prove_time
            self.last_metrics['status'] = 'submitted'
            # Note: prove_time from generate_zkp is just the snarkjs part, total_time includes eval etc.
        except Exception as e:
            # Catch errors from run_command within generate_zkp or other issues
            print(f"Client {self.client_id}: ZKP proof generation failed: {e}", file=sys.stderr)
            proof_result_tuple = None
            proof_size_bytes = 0
            # total_time is calculated finally

        total_time = time.time() - start_time
        print(f"Client {self.client_id}: Proof generation attempt took {total_time:.2f}s", file=sys.stderr)

        if proof_result_tuple:
            # proof_result_tuple is (proof_obj, generated_public_inputs_list)
            # We need to return the *original* public_inputs_list for the server verification
            proof_obj, generated_public_inputs = proof_result_tuple
            if generated_public_inputs != public_inputs_list:
                raise ValueError("Generated public signals differ from the requested statement")
            return (proof_obj, generated_public_inputs), proof_size_bytes, total_time
        else:
            print(f"Client {self.client_id}: Failed to generate ZKP proof.", file=sys.stderr)
            return None, 0, total_time
