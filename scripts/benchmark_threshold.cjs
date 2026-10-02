const fs=require('fs'),snarkjs=require('snarkjs');
const {performance}=require('perf_hooks');
async function main() {
 const dir='zkp_setup/threshold';
 const vk=JSON.parse(fs.readFileSync(dir+'/verification_key.json'));
 const input={model_hash:'12345678901234567890',threshold:'1000000',nonce:'1234567890123456789012345',calculated_loss:'0'};
 const rows=[];let example;
 for(let i=0;i<12;i++) {
  let t=performance.now();
  await snarkjs.wtns.calculate(input,dir+'/loss_threshold_js/loss_threshold.wasm',dir+'/bench.wtns');
  const witness_s=(performance.now()-t)/1000;t=performance.now();
  const result=await snarkjs.groth16.prove(dir+'/final.zkey',dir+'/bench.wtns',undefined,{singleThread:true});
  const prove_s=(performance.now()-t)/1000;t=performance.now();
  const valid=await snarkjs.groth16.verify(vk,result.publicSignals,result.proof);
  const verify_s=(performance.now()-t)/1000;
  if(!valid)throw Error('Threshold baseline rejected');
  if(i>=2)rows.push({witness_s,prove_s,verify_s,proof_json_bytes:Buffer.byteLength(JSON.stringify(result.proof)),
    public_json_bytes:Buffer.byteLength(JSON.stringify(result.publicSignals))});
  example=result;
 }
 const changedNonce=[...example.publicSignals];changedNonce[2]='457';
 const changedModel=[...example.publicSignals];changedModel[0]='124';
 const report={measurements:rows,guarantee:'supplied-loss only; zero fabricated loss accepted without evaluating a model',
  fabricated_loss_accepted:true,
  changed_nonce_accepted:await snarkjs.groth16.verify(vk,changedNonce,example.proof),
  changed_model_hash_accepted:await snarkjs.groth16.verify(vk,changedModel,example.proof)};
 fs.writeFileSync('results/threshold-baseline.json',JSON.stringify(report,null,2));
 fs.unlinkSync(dir+'/bench.wtns');
}
main().then(()=>process.exit(0)).catch(e=>{console.error(e);process.exit(1)});
