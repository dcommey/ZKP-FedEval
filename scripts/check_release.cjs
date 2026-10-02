// Test incomplete-cohort decisions, including selective withholding.
const assert=require('assert/strict'),fs=require('fs'),os=require('os'),path=require('path');
const {CohortReleasePolicy}=require('./protocol_policy.cjs');
const ids=Array.from({length:10},(_,i)=>`batch-${i}`);
const checks=[];
function check(name,run){run();checks.push({name,passed:true})}
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'fedeval-release-'));
let withholding;
try {
 check('selective_withholding_cannot_certify_release',()=>{
  const q=new CohortReleasePolicy('model-cutoff-round-1',ids,8);q.recordVerified(ids[0],'1');q.recordVerified(ids[1],'1');
  withholding=q.summary();assert.equal(withholding.respondent_positive_fraction,1);
  assert.equal(withholding.lower_bound,.2);assert.equal(withholding.upper_bound,1);
  assert.equal(withholding.unresolved,8);assert.equal(withholding.decision,'inconclusive');
 });
 check('complete_failing_cohort_rejected',()=>{
  const q=new CohortReleasePolicy('model-cutoff-round-1',ids,8);
  ids.forEach((id,i)=>q.recordVerified(id,i<2?'1':'0'));
  assert.equal(q.summary().decision,'certified_reject');assert.equal(q.summary().upper_bound,.2);
 });
 check('exact_required_count_certifies_release_with_unresolved_batches',()=>{
  const q=new CohortReleasePolicy('model-cutoff-round-1',ids,8);ids.slice(0,8).forEach(id=>q.recordVerified(id,'1'));
  assert.equal(q.summary().decision,'certified_release');assert.equal(q.summary().unresolved,2);
 });
 check('insufficient_known_passes_remain_inconclusive',()=>{
  const q=new CohortReleasePolicy('model-cutoff-round-1',ids,8);ids.slice(0,7).forEach(id=>q.recordVerified(id,'1'));
  ids.slice(7,9).forEach(id=>q.recordVerified(id,'0'));
  assert.equal(q.summary().decision,'inconclusive');
  q.recordVerified(ids[9],'0');assert.equal(q.summary().decision,'certified_reject');
 });
 check('empty_response_set_remains_inconclusive',()=>{
  const s=new CohortReleasePolicy('model-cutoff-round-1',ids,8).summary();
  assert.equal(s.respondent_positive_fraction,null);assert.equal(s.decision,'inconclusive');
 });
 check('invalid_or_unregistered_results_do_not_shrink_cohort',()=>{
  const q=new CohortReleasePolicy('model-cutoff-round-1',ids,8);
  assert.equal(q.recordVerified('outsider','1'),false);assert.equal(q.recordVerified(ids[0],'invalid'),false);
  assert.equal(q.summary().registered,10);assert.equal(q.summary().unresolved,10);
 });
 check('duplicate_or_changed_outcome_cannot_inflate_pass_count',()=>{
  const q=new CohortReleasePolicy('model-cutoff-round-1',ids,8);assert(q.recordVerified(ids[0],'0'));
  assert.equal(q.recordVerified(ids[0],'1'),false);assert.equal(q.summary().positive,0);
 });
 const ledger=path.join(tmp,'cohort.json');
 check('cohort_outcomes_and_finalization_survive_restart',()=>{
  const q=new CohortReleasePolicy('model-cutoff-round-1',ids,8,ledger);q.recordVerified(ids[0],'1');q.finalize();
  const restart=new CohortReleasePolicy('model-cutoff-round-1',ids,8,ledger);
  assert.equal(restart.summary().positive,1);assert.equal(restart.summary().finalized,true);
  assert.equal(restart.recordVerified(ids[1],'1'),false);
 });
 check('cohort_or_requirement_cannot_change_on_restart',()=>{
  assert.throws(()=>new CohortReleasePolicy('model-cutoff-round-1',ids,7,ledger));
  assert.throws(()=>new CohortReleasePolicy('model-cutoff-round-1',[...ids.slice(0,9),'replacement'],8,ledger));
 });
 check('different_model_or_round_request_cannot_reuse_outcome_ledger',()=>{
  assert.throws(()=>new CohortReleasePolicy('different-model-or-round',ids,8,ledger));
 });
 check('invalid_enrollment_or_release_requirement_refused',()=>{
  assert.throws(()=>new CohortReleasePolicy('model-cutoff-round-1',[],1));assert.throws(()=>new CohortReleasePolicy('model-cutoff-round-1',['a','a'],1));
  assert.throws(()=>new CohortReleasePolicy('model-cutoff-round-1',ids,0));assert.throws(()=>new CohortReleasePolicy('model-cutoff-round-1',ids,11));
 });
 check('all_small_cohort_decisions_agree_with_every_possible_completion',()=>{
  for(let N=1;N<=10;N++)for(let K=1;K<=N;K++)for(let p=0;p<=N;p++)for(let f=0;f<=N-p;f++){
   const cohort=Array.from({length:N},(_,i)=>String(i)),q=new CohortReleasePolicy('model-cutoff-round-1',cohort,K);
   cohort.slice(0,p).forEach(id=>q.recordVerified(id,'1'));
   cohort.slice(p,p+f).forEach(id=>q.recordVerified(id,'0'));
   const possible=[];for(let remainingPass=0;remainingPass<=N-p-f;remainingPass++)possible.push(p+remainingPass>=K);
   const expected=possible.every(Boolean)?'certified_release':possible.every(x=>!x)?'certified_reject':'inconclusive';
   assert.equal(q.summary().decision,expected,`N=${N} K=${K} p=${p} f=${f}`);
  }
 });
 fs.mkdirSync('results',{recursive:true});
 fs.writeFileSync('results/release-policy.json',JSON.stringify({checks,withholding_example:withholding,
  guarantee:'fixed registered-cohort integer release criterion; missing and invalid remain unresolved',
  precondition:'recordVerified receives only cryptographically verified outcomes matching the registered request; protected serialized ledger'},null,2));
 console.log(`${checks.length} release-policy checks passed`);
}finally{fs.rmSync(tmp,{recursive:true,force:true})}
