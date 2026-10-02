// Regression checks for privacy-policy bypasses and durable consumption.
const assert=require('assert/strict'),fs=require('fs'),os=require('os'),path=require('path');
const {ClientQueryPolicy,VerifierPolicy,modelDigest}=require('./protocol_policy.cjs');
async function main(){
 const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'fedeval-policy-'));
 const ledger=path.join(tmp,'ledger.json');
 const base={weights:Array.from({length:10},()=>Array(16).fill('0')),bias:Array(10).fill('0'),dataset_root:'101',threshold:'100'};
 const changed=JSON.parse(JSON.stringify(base));changed.weights[0][0]='1';
 const unapproved=JSON.parse(JSON.stringify(base));unapproved.weights[0][0]='2';
 const A=modelDigest(base.weights,base.bias),B=modelDigest(changed.weights,changed.bias);
 const enrollment={batch_id:'site/device/heldout-1',root:'101',approved_requests:[
  {model_digest:A,threshold:'100'}, {model_digest:B,threshold:'100'}]};
 const checks=[];
 const check=(name,fn)=>{assert.equal(fn(),true,name);checks.push({name,passed:true})};
 try {
  const q=new ClientQueryPolicy(enrollment,ledger);
  check('unapproved_model_refused',()=>!q.authorize({...unapproved,model_digest:A}));
  check('unapproved_cutoff_refused',()=>!q.authorize({...base,threshold:'101'}));
  check('unregistered_root_refused',()=>!q.authorize({...base,dataset_root:'102'}));
  check('first_approved_query_allowed',()=>q.authorize(base));
  check('same_model_repeat_refused',()=>!q.authorize(base));
  check('different_approved_model_same_batch_refused',()=>!q.authorize(changed));
  const restart=new ClientQueryPolicy(enrollment,ledger);
  check('different_model_after_restart_refused',()=>!restart.authorize(changed));
  const resalt=new ClientQueryPolicy({...enrollment,root:'202'},ledger);
  check('resalted_same_batch_refused',()=>!resalt.authorize({...base,dataset_root:'202'}));
  const next=new ClientQueryPolicy({...enrollment,batch_id:'site/device/heldout-2',root:'303'},ledger);
  check('new_enrolled_batch_allowed',()=>next.authorize({...base,dataset_root:'303'}));
  const mutable={...enrollment,batch_id:'immutable',approved_requests:[{model_digest:A,threshold:'100'}]};
  const isolated=new ClientQueryPolicy(mutable);
  mutable.root='evil';mutable.approved_requests.push({model_digest:'evil',threshold:'100'});
  check('configuration_mutation_cannot_add_model',()=>!isolated.authorize(unapproved));
  check('configuration_mutation_cannot_replace_root',()=>!isolated.authorize({...base,dataset_root:'evil'}));
  fs.writeFileSync(path.join(tmp,'legacy.json'),JSON.stringify(['A:101']));
  check('legacy_ledger_fails_closed',()=>{try{new ClientQueryPolicy(enrollment,path.join(tmp,'legacy.json'));return false}catch{return true}});
  const shared=path.join(tmp,'shared-query.json');
  const qa=new ClientQueryPolicy({...enrollment,batch_id:'alpha'},shared);
  const qb=new ClientQueryPolicy({...enrollment,batch_id:'beta'},shared);
  assert(qa.authorize(base));assert(qb.authorize(base));
  check('stale_client_instances_preserve_consumption',()=>
   !new ClientQueryPolicy({...enrollment,batch_id:'alpha'},shared).authorize(base)&&
   !new ClientQueryPolicy({...enrollment,batch_id:'beta'},shared).authorize(base));
  const signals=['context','1','client','round'];
  const v=new VerifierPolicy(signals,path.join(tmp,'verifier.json'));
  const outcomes=await Promise.all([v.accept(signals,async()=>true),v.accept(signals,async()=>true)]);
  check('concurrent_duplicate_refused',()=>outcomes.filter(Boolean).length===1);
  const vr=new VerifierPolicy(signals,path.join(tmp,'verifier.json'));
  check('verifier_duplicate_survives_restart',()=>vr.used.has(vr.key));
  const failed=new VerifierPolicy(['other','1','different','round']);
  const invalidAccepted=await failed.accept(['other','1','different','round'],async()=>false);
  check('invalid_proof_refused',()=>!invalidAccepted);
  const sharedV=path.join(tmp,'shared-verifier.json');
  const va=new VerifierPolicy(['ctxA','1','alpha','round'],sharedV);
  const vb=new VerifierPolicy(['ctxB','1','beta','round'],sharedV);
  assert(await va.accept(['ctxA','1','alpha','round'],async()=>true));
  assert(await vb.accept(['ctxB','1','beta','round'],async()=>true));
  const saved=new Set(JSON.parse(fs.readFileSync(sharedV)));
  check('stale_verifier_instances_preserve_acceptance',()=>saved.has('alpha:round')&&saved.has('beta:round'));
  fs.mkdirSync('results',{recursive:true});
  fs.writeFileSync('results/policy-checks.json',JSON.stringify({version:2,checks,
   guarantee:'one attempted evaluation per stable enrolled batch, across approved models and roots; serialized trusted client service',
   negative_checks:15,positive_controls:2},null,2));
  console.log(`${checks.length} policy checks passed`);
 }finally{fs.rmSync(tmp,{recursive:true,force:true})}
}
main().catch(e=>{console.error(e);process.exit(1)});
