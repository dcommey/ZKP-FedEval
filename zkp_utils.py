"""
Zero-Knowledge Proof (ZKP) utilities for federated evaluation.
Provides functions for ZKP setup, generation, and verification using Circom and SnarkJS.
"""

import subprocess
import json
import os
import time
import hashlib
import numpy as np
import math
import sys
import re
import shutil # Import shutil for directory removal

# Directory and file configuration
ZKP_ARTIFACTS_DIR = './zkp_artifacts'
CIRCUIT_DIR = './circuits'
CIRCUIT_NAME = 'loss_threshold'
CIRCUIT_PATH = os.path.join(CIRCUIT_DIR, f"{CIRCUIT_NAME}.circom")
# Use this constant for the Powers of Tau file path
PTAU_FILE = os.path.join(ZKP_ARTIFACTS_DIR, "pot12_final.ptau") # Corrected filename

# Circuit parameter configuration
FIXED_POINT_SCALE_LOG10 = 6  # 6 decimal places of precision
FIXED_POINT_SCALE = 10**FIXED_POINT_SCALE_LOG10
CIRCUIT_PRECISION_BITS = 64  # Maximum bits for fixed-point representation

# Define SCALE_FACTOR based on environment variable, default to 1e6 (6 decimal places)
try:
    ZKP_PRECISION = int(os.environ.get('ZKP_PRECISION', 6))
    if ZKP_PRECISION < 1:
        print("Warning: ZKP_PRECISION must be at least 1. Using 6.", file=sys.stderr)
        ZKP_PRECISION = 6
except ValueError:
    print("Warning: Invalid ZKP_PRECISION value. Using 6.", file=sys.stderr)
    ZKP_PRECISION = 6
SCALE_FACTOR = 10**ZKP_PRECISION
print(f"Using ZKP Precision: {ZKP_PRECISION} decimal places (Scale Factor: {SCALE_FACTOR})", file=sys.stderr)

def run_command(cmd, cwd=None):
    """
    Execute a shell command and return its output.
    Raises subprocess.CalledProcessError if command fails, including stderr in the message.
    """
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        # Include stderr in the exception message for better debugging
        error_message = (
            f"Command '{' '.join(cmd)}' returned non-zero exit status {result.returncode}.\n"
            f"Stderr:\n{result.stderr}\n"
            f"Stdout:\n{result.stdout}"
        )
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr
        ) # Consider raising with the custom message if needed, but the default includes stderr too.
        # The default CalledProcessError includes stderr, so just raising it might be enough.
        # Let's stick to the standard raise for now, the calling function prints the exception.
        # The key is that the exception *contains* stderr.
    # Return the completed process object on success
    return result # Return the result object which contains stdout and stderr

def to_fixed_point(value):
    """
    Convert float to fixed-point representation suitable for ZKP circuit.
    Warns if value exceeds circuit's precision range.
    """
    scaled_value = value * FIXED_POINT_SCALE
    max_val = 2**(CIRCUIT_PRECISION_BITS - 1) - 1
    min_val = -2**(CIRCUIT_PRECISION_BITS - 1)
    if not (min_val <= scaled_value <= max_val):
        print(f"Warning: Value {value} scaled to {scaled_value} might be outside expected {CIRCUIT_PRECISION_BITS}-bit range [{min_val}, {max_val}]", file=sys.stderr)
    return str(int(round(scaled_value)))

def string_to_int(s, max_bits=253):
    """
    Hash a string to an integer suitable for ZKP circuit input.
    Uses SHA-256 and ensures output fits within field size.
    """
    hasher = hashlib.sha256(s.encode('utf-8'))
    hex_hash = hasher.hexdigest()
    int_val = int(hex_hash, 16)
    max_val_field = (1 << max_bits) - 1
    truncated_int = int_val & max_val_field
    return str(truncated_int)

def setup_zkp(circuit_path="circuits/loss_threshold.circom", setup_dir="zkp_setup"):
    """
    Sets up the ZKP environment (compile circuit, generate keys).
    Uses the PTAU_FILE constant for the Powers of Tau file.
    Returns a dictionary containing paths to keys, constraint count, and witness calculator.
    """
    print("Starting ZKP setup...", file=sys.stderr)

    # --- Clean previous setup directory ---
    if os.path.exists(setup_dir):
        print(f"Removing existing setup directory: {setup_dir}", file=sys.stderr)
        try:
            shutil.rmtree(setup_dir)
        except OSError as e:
            print(f"Warning: Could not remove existing setup directory {setup_dir}: {e}", file=sys.stderr)
            # Decide if this should be a fatal error or just a warning
            # Forcing a clean state is safer, so let's make it fatal if removal fails significantly
            if os.path.exists(setup_dir): # Check if it still exists after attempted removal
                 raise RuntimeError(f"Failed to remove existing setup directory {setup_dir}. Cannot guarantee clean setup.") from e
    # --- End Clean ---

    # Ensure necessary directories exist
    os.makedirs(setup_dir, exist_ok=True)
    os.makedirs(ZKP_ARTIFACTS_DIR, exist_ok=True) # Ensure artifacts dir exists for PTAU_FILE

    circuit_name = os.path.splitext(os.path.basename(circuit_path))[0]
    r1cs_path = os.path.join(setup_dir, f"{circuit_name}.r1cs")
    wasm_path = os.path.join(setup_dir, f"{circuit_name}_js", f"{circuit_name}.wasm")
    pkey_path = os.path.join(setup_dir, f"{circuit_name}_final.zkey")
    vkey_path = os.path.join(setup_dir, f"{circuit_name}_verification_key.json")
    js_dir = os.path.join(setup_dir, f"{circuit_name}_js")
    # Define path for the witness calculator script generated by circom
    witness_calculator_path = os.path.join(js_dir, "generate_witness.js")

    constraint_count = -1

    try:
        # 1. Compile the circuit using Circom
        print(f"Compiling circuit: {circuit_path}...", file=sys.stderr)

        # Corrected compile command: specify output directory only once AND add library path
        compile_command = [
            "circom", circuit_path,
            "--r1cs",   # Generate R1CS
            "--wasm",   # Generate WASM
            "--sym",    # Generate SYM
            "-l", "./node_modules", # Add path to node_modules for circomlib
            "-o", setup_dir # Specify the output directory ONCE
        ]
        result = run_command(compile_command)
        stdout = result.stdout
        stderr = result.stderr
        print(stdout, file=sys.stderr) # Print circom output
        if stderr:
            # Print warnings even if the command succeeded overall
            print(f"Circom compilation stderr (warnings/info):\n{stderr}", file=sys.stderr)

        # Parse constraint count from stdout
        match = re.search(r"Constraints: (\d+)", stdout) # More general regex
        if not match:
             match = re.search(r"non-linear constraints: (\d+)", stdout) # Fallback for older circom?
        if match:
            constraint_count = int(match.group(1))
            print(f"Parsed constraint count: {constraint_count}", file=sys.stderr)
        else:
            print("Warning: Could not parse constraint count from circom output.", file=sys.stderr)
            # Optionally raise an error if constraints are mandatory
            # raise ValueError("Failed to parse constraint count from circom output.")


        # Check if R1CS and WASM files were created
        if not os.path.exists(r1cs_path):
            raise FileNotFoundError(f"R1CS file not found after compilation: {r1cs_path}")
        if not os.path.exists(wasm_path):
             # Try alternative common path if the first fails
             wasm_path_alt = os.path.join(setup_dir, f"{circuit_name}.wasm")
             if os.path.exists(wasm_path_alt):
                 wasm_path = wasm_path_alt
                 print(f"Note: WASM file found at alternative path: {wasm_path}", file=sys.stderr)
             else:
                 raise FileNotFoundError(f"WASM file not found after compilation in {js_dir} or {setup_dir}")

        # Check if witness calculator script exists
        if not os.path.exists(witness_calculator_path):
             # Check alternative path if first fails (older circom might place it differently)
             witness_calculator_path_alt = os.path.join(setup_dir, "generate_witness.js")
             if os.path.exists(witness_calculator_path_alt):
                  witness_calculator_path = witness_calculator_path_alt
                  print(f"Note: Witness calculator found at alternative path: {witness_calculator_path}", file=sys.stderr)
             else:
                  raise FileNotFoundError(f"Witness calculator script not found after compilation in {js_dir} or {setup_dir}")

        # 2. Generate witness (Example - typically done by client, but setup needs wasm)
        # We don't generate a witness here, just ensure wasm exists for keygen

        # 3. Check for Powers of Tau file using the PTAU_FILE constant
        # Use the globally defined PTAU_FILE path
        ptau_path = PTAU_FILE
        if not os.path.exists(ptau_path):
             # Removed download logic. Now it's an error if the file isn't where expected.
             print(f"Error: Powers of Tau file not found at the expected location: {ptau_path}", file=sys.stderr)
             print("Please ensure the file exists. You might need to download it manually or adjust the PTAU_FILE path in zkp_utils.py.", file=sys.stderr)
             raise FileNotFoundError(f"Required Powers of Tau file not found: {ptau_path}")
        else:
             print(f"Using Powers of Tau file: {ptau_path}", file=sys.stderr)


        # 4. Generate Phase 2 keys (zkey)
        print("Generating ZKP keys (Phase 2)...", file=sys.stderr)
        zkey_initial_path = os.path.join(setup_dir, f"{circuit_name}_0000.zkey")
        # snarkjs groth16 setup command now uses the verified ptau_path
        run_command([
            "snarkjs", "groth16", "setup", r1cs_path, ptau_path, zkey_initial_path
        ])

        # Contribute to Phase 2 (single local contribution for reproducibility).
        run_command([
            "snarkjs", "zkey", "contribute", zkey_initial_path, pkey_path,
            "-v", "-e=\"some random text\""
        ])

        # Export verification key
        run_command([
            "snarkjs", "zkey", "export", "verificationkey", pkey_path, vkey_path
        ])

        print("ZKP setup completed successfully.", file=sys.stderr)
        # Return paths including the witness calculator and using consistent key names
        return {
            "circuit_path": circuit_path,
            "setup_dir": setup_dir,
            "r1cs_path": r1cs_path,
            "wasm_path": wasm_path, # Use this key for wasm file
            "pkey_path": pkey_path, # Use this key for proving key
            "vkey_path": vkey_path, # Use this key for verification key
            "witness_calculator_path": witness_calculator_path, # Use this key for witness calculator
            "constraint_count": constraint_count
        }

    except FileNotFoundError as e:
        print(f"Error during ZKP setup: Required file not found. {e}", file=sys.stderr)
        raise e
    except subprocess.CalledProcessError as e:
        # This will now catch the error from run_command
        # The exception 'e' itself contains stdout and stderr
        print(f"An error occurred during ZKP setup (running command): {e}", file=sys.stderr)
        # Explicitly print stderr if it's not automatically shown by the default exception string
        if e.stderr:
             print(f"Command stderr:\n{e.stderr}", file=sys.stderr)
        raise e # Re-raise the original exception
    except Exception as e:
        print(f"An error occurred during ZKP setup: {e}", file=sys.stderr)
        # Consider printing stdout/stderr from run_command if available and relevant
        raise e

def generate_zkp(setup_info, public_inputs_dict, private_inputs_dict):
    """
    Generate a ZKP proof using the provided inputs and setup information.
    
    Args:
        setup_info: Dictionary containing paths to required ZKP artifacts
                    (expects 'witness_calculator_path', 'wasm_path', 'pkey_path')
        public_inputs_dict: Public inputs visible to verifier
        private_inputs_dict: Private witness values
    
    Returns:
        tuple: ((proof, public_inputs), proof_size_bytes, elapsed_time)
    """
    print("Generating ZKP proof...", file=sys.stderr)
    timestamp = time.time_ns()
    
    # Temporary file paths
    input_file = os.path.join(ZKP_ARTIFACTS_DIR, f"input_{timestamp}.json")
    witness_file = os.path.join(ZKP_ARTIFACTS_DIR, f"witness_{timestamp}.wtns")
    proof_file = os.path.join(ZKP_ARTIFACTS_DIR, f"proof_{timestamp}.json")
    public_outputs_file = os.path.join(ZKP_ARTIFACTS_DIR, f"public_{timestamp}.json")

    start_time = time.time()
    proof_size_bytes = 0
    proof = None
    generated_public_inputs = None

    try:
        combined_inputs = {**public_inputs_dict, **private_inputs_dict}
        with open(input_file, 'w') as f:
            json.dump(combined_inputs, f, indent=2)
        print(f"Input file created: {input_file}", file=sys.stderr)

        # Do not print combined inputs here (may include private witness values).

        print("Calculating witness...", file=sys.stderr)
        # Use correct keys from setup_info as defined in setup_zkp return dict
        wit_cmd = [
            'node', setup_info['witness_calculator_path'], # Use correct key
            setup_info['wasm_path'],                     # Use correct key
            input_file,
            witness_file
        ]
        run_command(wit_cmd) # This might raise CalledProcessError
        print("Witness calculated.", file=sys.stderr)

        print("Generating Groth16 proof...", file=sys.stderr)
        # Use correct keys from setup_info as defined in setup_zkp return dict
        prove_cmd = [
            'snarkjs', 'groth16', 'prove',
            setup_info['pkey_path'], # Use correct key (proving key)
            witness_file,
            proof_file,
            public_outputs_file
        ]
        run_command(prove_cmd) # This might raise CalledProcessError
        print("Proof generated.", file=sys.stderr)

        with open(proof_file, 'r') as f: proof = json.load(f)
        with open(public_outputs_file, 'r') as f: generated_public_inputs = json.load(f)
        proof_size_bytes = os.path.getsize(proof_file)

    except subprocess.CalledProcessError as e:
        # Explicitly print stderr for CalledProcessError
        print(f"ERROR during ZKP generation step (running command): {e}", file=sys.stderr)
        if e.stderr:
             print(f"Command stderr:\n{e.stderr}", file=sys.stderr)
        if e.stdout: # Also print stdout for context
             print(f"Command stdout:\n{e.stdout}", file=sys.stderr)
        # Cleanup temp files
        for fpath in [input_file, witness_file, proof_file, public_outputs_file]:
            try:
                if os.path.exists(fpath): os.remove(fpath)
            except OSError: pass
        raise # Re-raise the exception
    except Exception as e:
        # Catch other potential errors (file I/O, etc.)
        print(f"ERROR during ZKP generation steps: {e}", file=sys.stderr)
        # Cleanup temp files
        for fpath in [input_file, witness_file, proof_file, public_outputs_file]:
            try:
                if os.path.exists(fpath): os.remove(fpath)
            except OSError: pass
        raise # Re-raise the exception

    finally:
        try:
            if os.path.exists(input_file): os.remove(input_file)
            if os.path.exists(witness_file): os.remove(witness_file)
        except OSError as e:
            print(f"Warning: Could not clean up intermediate files: {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    print(f"Proof generation completed in {elapsed:.2f} seconds. Proof size: {proof_size_bytes / 1024:.2f} KiB", file=sys.stderr)
    return (proof, generated_public_inputs), proof_size_bytes, elapsed

def verify_zkp(setup_info, public_inputs_list, proof):
    """
    Verify a ZKP proof against provided public inputs.
    
    Args:
        setup_info: Dictionary containing paths to required ZKP artifacts
                    (expects 'vkey_path')
        public_inputs_list: List of public inputs to verify against
        proof: The proof object to verify
    
    Returns:
        tuple: (is_valid, elapsed_time)
    """
    print("Verifying ZKP proof...", file=sys.stderr)
    timestamp = time.time_ns()
    
    # Temporary files for verification
    temp_proof_file = os.path.join(ZKP_ARTIFACTS_DIR, f"verify_proof_{timestamp}.json")
    temp_public_file = os.path.join(ZKP_ARTIFACTS_DIR, f"verify_public_{timestamp}.json")

    start_time = time.time()
    is_valid = False

    try:
        with open(temp_proof_file, 'w') as f: json.dump(proof, f)
        with open(temp_public_file, 'w') as f: json.dump(public_inputs_list, f)

        # Use correct key from setup_info as defined in setup_zkp return dict
        verify_cmd = [
            'snarkjs', 'groth16', 'verify',
            setup_info['vkey_path'], # Use correct key (verification key)
            temp_public_file,
            temp_proof_file
        ]
        result = run_command(verify_cmd)
        if "OK!" in result.stdout:
            is_valid = True
            print("Verification successful (OK!).", file=sys.stderr)
        else:
            print(f"Verification failed (but command succeeded?). Snarkjs output:\n{result.stdout}", file=sys.stderr)
            is_valid = False

    except subprocess.CalledProcessError as e:
        if "Invalid proof" in e.stderr:
            print("Verification failed (Invalid proof).", file=sys.stderr)
        else:
            # Print more details when the command fails for other reasons
            print("Verification command failed.", file=sys.stderr)
            if e.stderr:
                 print(f"Command stderr:\n{e.stderr}", file=sys.stderr)
            if e.stdout: # Also print stdout for context
                 print(f"Command stdout:\n{e.stdout}", file=sys.stderr)
        is_valid = False

    except Exception as e:
        print(f"An unexpected error occurred during verification: {e}", file=sys.stderr)
        is_valid = False

    finally:
        try:
            if os.path.exists(temp_proof_file): os.remove(temp_proof_file)
            if os.path.exists(temp_public_file): os.remove(temp_public_file)
        except OSError as e:
            print(f"Warning: Could not clean up verification temp files: {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    print(f"Verification completed in {elapsed:.2f} seconds. Result: {is_valid}", file=sys.stderr)
    return is_valid, elapsed

def prepare_circuit_inputs(local_loss, threshold, nonce, model_hash):
    """
    Prepares the public and private inputs for the loss_threshold circuit.

    Args:
        local_loss (float): The client's computed loss.
        threshold (float): The loss threshold.
        nonce (str): The round-specific nonce.
        model_hash (str): The hash of the model weights.

    Returns:
        tuple: (public_inputs_list, public_inputs_dict, private_inputs_dict)
               Returns None, None, None if conversion fails.
    """
    try:
        # Scale and convert floats to integers
        scaled_loss_str = str(int(local_loss * SCALE_FACTOR))
        scaled_threshold_str = str(int(threshold * SCALE_FACTOR))

        # Convert hex model hash and nonce to integer representation suitable for circuit
        # Assuming model_hash is hex string, nonce is alphanumeric
        # Note: This repository uses a lightweight placeholder encoding to map
        # (model_hash, nonce) into field elements. It is not intended to provide
        # strong collision resistance. A production design should use an in-circuit
        # hash (e.g., Poseidon) or a well-specified binding scheme.

        # Simplified: Use first 16 chars of hash and nonce for smaller integer inputs
        # Convert hex hash part to int
        hash_int_str = str(int(model_hash[:16], 16))
        # Convert nonce part (base36) to int - ensure nonce is suitable
        nonce_int_str = str(int(nonce[:16], 36)) # Assuming nonce fits base36 conversion


        # Public inputs list (order matters for verification)
        # --- CORRECT ORDER HERE ---
        # Must match the order defined in the circuit/sym file: [model_hash, threshold, nonce]
        public_inputs_list = [
            hash_int_str,          # Index 0 (Corresponds to witness index 1)
            scaled_threshold_str,  # Index 1 (Corresponds to witness index 2)
            nonce_int_str          # Index 2 (Corresponds to witness index 3)
        ]
        # --- END CORRECTION ---

        # Public inputs dictionary (for input.json generation - order doesn't matter here)
        # Keys must match signal names in the circuit
        public_inputs_dict = {
            "threshold": scaled_threshold_str,
            "nonce": nonce_int_str,
            "model_hash": hash_int_str
        }

        # Private inputs dictionary
        private_inputs_dict = {
            "calculated_loss": scaled_loss_str # Use the name from the .sym file
        }

        # Return list for verification, dict for generation, dict for generation
        return public_inputs_list, public_inputs_dict, private_inputs_dict

    except ValueError as e:
        print(f"Error converting inputs for circuit: {e}", file=sys.stderr)
        print(f"  local_loss={local_loss}, threshold={threshold}, nonce={nonce}, model_hash={model_hash}", file=sys.stderr)
        return None, None, None
    except Exception as e:
        print(f"Unexpected error preparing circuit inputs: {e}", file=sys.stderr)
        return None, None, None

def hash_model_weights(model):
    """
    Compute SHA-256 hash of model weights.
    
    Args:
        model: PyTorch model
    
    Returns:
        str: Hexadecimal hash of the model weights
    """
    hasher = hashlib.sha256()
    state_dict = model.state_dict()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode('utf-8'))
        tensor_bytes = state_dict[key].cpu().numpy().tobytes()
        hasher.update(tensor_bytes)
    hex_digest = hasher.hexdigest()
    return hex_digest