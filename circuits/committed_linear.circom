pragma circom 2.2.3;
include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/comparators.circom";

// Entire, ordered, pre-registered batch; no prover-selected subset.
// Signed inputs/weights use field encodings with explicit integer bounds.
template CommittedLinear(B, D, C) {
    signal input weights[C][D];
    signal input bias[C];
    signal input threshold;
    signal input dataset_root;
    signal input client_id;
    signal input nonce;
    signal input features[B][D];
    signal input labels[B][C];
    signal input salt;
    signal output context_tag;
    signal output passed;

    component context = Poseidon(4);
    context.inputs[0] <== dataset_root;
    context.inputs[1] <== client_id;
    context.inputs[2] <== nonce;
    context.inputs[3] <== threshold;
    context_tag <== context.out;
    component saltRange = Num2Bits(128);
    saltRange.in <== salt;
    component initial = Poseidon(2);
    initial.inputs[0] <== client_id;
    initial.inputs[1] <== salt;
    signal roots[B+1];
    roots[0] <== initial.out;
    component featureHash[B];
    component sampleHash[B];
    component rootHash[B];
    component featureRange[B][D];
    component weightRange[C][D];
    component biasRange[C];
    signal products[B][C][D];
    signal scores[B][C];
    signal errors[B][C];
    signal squares[B][C];
    signal sum_loss;
    var total = 0;
    for (var c=0; c<C; c++) {
        biasRange[c] = Num2Bits(36);
        biasRange[c].in <== bias[c] + 34359738368;
        for (var d=0; d<D; d++) {
            weightRange[c][d] = Num2Bits(16);
            weightRange[c][d].in <== weights[c][d] + 32768;
        }
    }
    for (var i=0; i<B; i++) {
        featureHash[i] = Poseidon(D);
        for (var d=0; d<D; d++) {
            featureRange[i][d] = Num2Bits(16);
            featureRange[i][d].in <== features[i][d] + 32768;
            featureHash[i].inputs[d] <== features[i][d];
        }
        sampleHash[i] = Poseidon(C+2);
        sampleHash[i].inputs[0] <== featureHash[i].out;
        sampleHash[i].inputs[1] <== i;
        var labelSum = 0;
        for (var c=0; c<C; c++) {
            labels[i][c] * (labels[i][c]-1) === 0;
            labelSum += labels[i][c];
            sampleHash[i].inputs[c+2] <== labels[i][c];
            var score = bias[c];
            for (var d=0; d<D; d++) {
                products[i][c][d] <== weights[c][d] * features[i][d];
                score += products[i][c][d];
            }
            scores[i][c] <== score;
            errors[i][c] <== scores[i][c] - 10000*labels[i][c];
            squares[i][c] <== errors[i][c]*errors[i][c];
            total += squares[i][c];
        }
        labelSum === 1;
        rootHash[i] = Poseidon(3);
        rootHash[i].inputs[0] <== roots[i];
        rootHash[i].inputs[1] <== sampleHash[i].out;
        rootHash[i].inputs[2] <== client_id;
        roots[i+1] <== rootHash[i].out;
    }
    roots[B] === dataset_root;
    sum_loss <== total;
    // Bounds above imply the integer sum is below 2^80 (well below BN254).
    component lossRange = Num2Bits(80);
    component thresholdRange = Num2Bits(80);
    lossRange.in <== sum_loss;
    thresholdRange.in <== threshold;
    component below = LessThan(80);
    below.in[0] <== sum_loss;
    below.in[1] <== threshold;
    passed <== below.out;
}

component main {public [weights,bias,threshold,dataset_root,client_id,nonce]} = CommittedLinear(8,16,10);
