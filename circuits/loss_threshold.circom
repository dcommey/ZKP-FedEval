pragma circom 2.0.0;

// Use path relative to the include path (-l ./node_modules)
include "circomlib/circuits/comparators.circom"; 

/*
 * Verifies that a given private loss value is less than a public threshold.
 * Also includes public model hash and nonce placeholders.
 *
 * WARNING: This circuit DOES NOT verify the loss calculation itself.
 * It trusts the prover provided the correct 'calculated_loss'.
 * A production circuit needs to perform the NN forward pass internally.
 */
template LossThresholdVerifier(precision_bits) { // e.g., 64 bits for fixed-point representation
    // --- Public Inputs ---
    // These define the public statement being proved
    signal input model_hash;  // Hash of the model weights (passed as public input, not verified internally here)
                              // In a real circuit, this would be compared against a hash computed from private weights.
                              // For simplicity, treat as a large number/field element. Needs careful handling.
    signal input threshold;   // Loss threshold (in fixed-point representation)
    signal input nonce;       // Round-specific nonce (passed as public input, not verified internally here)
                              // Treat as a large number/field element.

    // --- Private Inputs ---
    // This is the witness provided by the prover
    signal input calculated_loss; // Loss computed by the client (in fixed-point representation)

    // --- Constraints ---
    // Verify: calculated_loss < threshold
    // Use LessThan component from circomlib
    component lt = LessThan(precision_bits);
    component lossRange = Num2Bits(precision_bits);
    component thresholdRange = Num2Bits(precision_bits);
    lossRange.in <== calculated_loss;
    thresholdRange.in <== threshold;
    lt.in[0] <== calculated_loss;
    lt.in[1] <== threshold;

    // Enforce the result of the comparison is 1 (true)
    // lt.out outputs 1 if in[0] < in[1], 0 otherwise.
    lt.out === 1;

    // --- Placeholders for future extensions ---
    // In a full circuit, you would add:
    // signal input private model_weights[...]
    // signal input private data_samples[...]
    // signal input private labels[...]
    // component nn_forward_pass = NNModel(...) // Complex NN logic
    // component loss_calc = LossFunction(...)
    // component hasher = Poseidon(...) // Hash the private weights
    // hasher.out === model_hash // Compare internal hash with public hash
    // loss_calc.out === calculated_loss // Ensure loss matches internal calculation
    // Include nonce in hash or computation to prevent replays...
}

// Instantiate the main component
// Choose precision (e.g., 64 bits). This affects the PTau file requirement.
// Public inputs MUST be listed in the order they should appear in the proof's public signals array.
component main { public [model_hash, threshold, nonce] } = LossThresholdVerifier(64);
