// Persistent-process measurements: excludes Node startup, includes witness work.
const fs = require('fs');
const {performance} = require('perf_hooks');
const snarkjs = require('snarkjs');
const {buildPoseidon} = require('circomlibjs');
const {VerifierPolicy,ClientQueryPolicy,modelDigest}=require('./protocol_policy.cjs');
const FIELD = 21888242871839275222246405745257275088548364400416034343698204186575808495617n;
const field = x => ((BigInt(x)%FIELD)+FIELD)%FIELD;
const clone = x => JSON.parse(JSON.stringify(x));
async function main() {
 const req=JSON.parse(fs.readFileSync(process.argv[2]));
 const poseidon=await buildPoseidon();
 const hash = a => poseidon.F.toObject(poseidon(a.map(field))).toString();
 const root = item => {
  let acc=hash([item.client_id,item.salt]);
  item.features.forEach((row,i) => {
   const fh=hash(row);
   const sh=hash([fh,i,...item.labels[i]]);
   acc=hash([acc,sh,item.client_id]);
  });
  return acc;
 };
 if(req.op==='enroll') {
  fs.writeFileSync(process.argv[3],JSON.stringify(req.items.map(root))); return;
 }
 const wasm=req.wasm, zkey=req.zkey;
 const vk=JSON.parse(fs.readFileSync(req.vkey));
 const integerLoss = input => input.features.reduce((sum,row,i)=>sum+input.weights.reduce((s,w,c)=> {
  const score=w.reduce((v,a,d)=>v+BigInt(a)*BigInt(row[d]),BigInt(input.bias[c]));
  const error=score-10000n*BigInt(input.labels[i][c]);return s+error*error;
 },0n),0n);
 const expected = input => [hash([input.dataset_root,input.client_id,input.nonce,input.threshold]),
   integerLoss(input)<BigInt(input.threshold)?'1':'0',
   ...input.weights.flat().map(x=>field(x).toString()),...input.bias.map(x=>field(x).toString()),
   input.threshold.toString(),input.dataset_root.toString(),input.client_id.toString(),input.nonce.toString()];
 const prove = async input => {
  const witness=process.argv[3]+'.wtns';
  let t=performance.now();
  await snarkjs.wtns.calculate(input,wasm,witness);
  const witness_s=(performance.now()-t)/1000;
  t=performance.now();
  const {proof,publicSignals}=await snarkjs.groth16.prove(zkey,witness,undefined,{singleThread:true});
  const prove_s=(performance.now()-t)/1000;
  if(JSON.stringify(publicSignals)!==JSON.stringify(expected(input))) throw Error('Public statement mismatch');
  t=performance.now();
  const valid=await snarkjs.groth16.verify(vk,publicSignals,proof);
  const verify_s=(performance.now()-t)/1000;
  if(!valid) throw Error('Honest proof rejected');
  fs.unlinkSync(witness);
  return {proof,publicSignals,witness_s,prove_s,verify_s,
   proof_json_bytes:Buffer.byteLength(JSON.stringify(proof)),
   public_json_bytes:Buffer.byteLength(JSON.stringify(publicSignals))};
 };
 const output=[];
 for(const task of req.tasks||[]) {
  const t=await prove(task.input);
  const {proof,publicSignals,...metrics}=t;
  output.push({...task.meta,...metrics});
  fs.writeFileSync(process.argv[3],JSON.stringify(output,null,2));
  console.log(`Completed ${output.length}/${req.tasks.length} ${task.meta.dataset} n=${task.meta.num_clients} T=${task.meta.threshold}`);
 }
 if(req.op==='security') {
  const input=req.input;
  const honest=await prove(input);
  const checks=[{name:'honest_proof',passed:true}];
  const witnessFails=async(name,mutate)=>{
   const bad=clone(input);mutate(bad);
   let rejected=false;
   try {await snarkjs.wtns.calculate(bad,wasm,process.argv[3]+'.attack.wtns');} catch(e) {rejected=true;}
   if(fs.existsSync(process.argv[3]+'.attack.wtns'))fs.unlinkSync(process.argv[3]+'.attack.wtns');
   checks.push({name,passed:rejected});
  };
  await witnessFails('changed_feature',x=>{x.features[0][0]=(BigInt(x.features[0][0])+1n).toString()});
  await witnessFails('changed_label',x=>{x.labels[0].fill('0'); x.labels[0][(input.labels[0].indexOf('1')+1)%x.labels[0].length]='1'});
  await witnessFails('duplicate_sample',x=>{x.features[1]=x.features[0];x.labels[1]=x.labels[0]});
  await witnessFails('reordered_batch',x=>{[x.features[0],x.features[1]]=[x.features[1],x.features[0]];[x.labels[0],x.labels[1]]=[x.labels[1],x.labels[0]]});
  await witnessFails('fabricated_zero_batch',x=>{x.features=x.features.map(r=>r.map(()=> '0'))});
  await witnessFails('non_one_hot_label',x=>{x.labels[0].fill('0');x.dataset_root=root(x)});
  await witnessFails('non_binary_label',x=>{x.labels[0].fill('0');x.labels[0][0]='2';x.labels[0][1]='-1';x.dataset_root=root(x)});
  await witnessFails('out_of_range_feature',x=>{x.features[0][0]='32768';x.dataset_root=root(x)});
  await witnessFails('out_of_range_weight',x=>{x.weights[0][0]='32768'});
  await witnessFails('out_of_range_threshold',x=>{x.threshold=(1n<<80n).toString()});
  for(const [name,cutoff,bit] of [['threshold_equality',req.loss_sum,'0'],
    ['threshold_below_loss',(BigInt(req.loss_sum)-1n).toString(),'0']]) {
   const boundary=clone(input);boundary.threshold=cutoff;
   const result=await prove(boundary);
   checks.push({name,passed:result.publicSignals[1]===bit});
  }
  for(const [name,mutate] of [
    ['wrong_client',x=>{x.client_id='99999'}], ['wrong_round',x=>{x.nonce='99999'}],
    ['wrong_dataset_root',x=>{x.dataset_root=(BigInt(x.dataset_root)+1n).toString()}],
    ['wrong_model',x=>{x.weights[0][0]=(BigInt(x.weights[0][0])+1n).toString()}],
    ['wrong_threshold',x=>{x.threshold=(BigInt(x.threshold)+1n).toString()}]]) {
   const altered=clone(input); mutate(altered);
   const accepted=await snarkjs.groth16.verify(vk,expected(altered),honest.proof);
   checks.push({name,passed:!accepted});
  }
  // Verifier policy depends only on the registered public statement, never data.
  const registered=expected(input);
  const policy=new VerifierPolicy(registered);
  const accept = signals => policy.accept(signals,s=>snarkjs.groth16.verify(vk,s,honest.proof));
  const forged=clone(input);forged.features[0][0]='0';forged.dataset_root=root(forged);
  checks.push({name:'client_reregistration_rejected',passed:!await accept(expected(forged))});
  checks.push({name:'registered_submission_accepted',passed:await accept(honest.publicSignals)});
  checks.push({name:'duplicate_submission_rejected',passed:!await accept(honest.publicSignals)});
  const budgetPath=process.argv[3]+'.budget.json';
  if(fs.existsSync(budgetPath))fs.unlinkSync(budgetPath);
  const enrollment={batch_id:'registered-batch',root:input.dataset_root,approved_requests:[{model_digest:modelDigest(input.weights,input.bias),threshold:input.threshold}]};
  const queries=new ClientQueryPolicy(enrollment,budgetPath);
  checks.push({name:'unagreed_threshold_refused',passed:!queries.authorize({...input,threshold:'2'})});
  checks.push({name:'first_agreed_query_allowed',passed:queries.authorize(input)});
  checks.push({name:'repeated_query_refused',passed:!queries.authorize(input)});
  const restarted=new ClientQueryPolicy(enrollment,budgetPath);
  checks.push({name:'query_budget_survives_restart',passed:!restarted.authorize(input)});
  fs.unlinkSync(budgetPath);
  const scale=[];
  for(const n of req.verify_counts||[]) {
   const t=performance.now();
   for(let i=0;i<n;i++) await snarkjs.groth16.verify(vk,honest.publicSignals,honest.proof);
   scale.push({proof_checks:n,total_s:(performance.now()-t)/1000,
    workload:'repeated valid proof; verifier throughput only, not distinct clients'});
  }
  fs.writeFileSync(process.argv[3],JSON.stringify({checks,honest_metrics:honest,verification_workload:scale},null,2));
 }
}
main().then(()=>process.exit(0)).catch(e=>{console.error(e);process.exit(1)});
