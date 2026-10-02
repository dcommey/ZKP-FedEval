// Application policy around the cryptographic relation. Identity authentication
// and registrar authorization are provided by the deployment, not by this class.
const fs=require('fs'),crypto=require('crypto');
const FIELD=21888242871839275222246405745257275088548364400416034343698204186575808495617n;
function modelDigest(weights,bias){
 if(!Array.isArray(weights)||weights.length!==10||weights.some(r=>!Array.isArray(r)||r.length!==16)||!Array.isArray(bias)||bias.length!==10)throw Error('Invalid model shape');
 const canonical=x=>{if(typeof x!=='string'||!/^[-]?\d+$/.test(x))throw Error('Invalid model integer');return ((BigInt(x)%FIELD+FIELD)%FIELD).toString()};
 return crypto.createHash('sha256').update(JSON.stringify({weights:weights.map(r=>r.map(canonical)),bias:bias.map(canonical)})).digest('hex');
}

function readState(path) {return new Set(path&&fs.existsSync(path)?JSON.parse(fs.readFileSync(path)):[])}
function saveState(path,state) {
 if(!path)return;
 fs.writeFileSync(path+'.tmp',JSON.stringify([...state]));fs.renameSync(path+'.tmp',path);
}
class VerifierPolicy {
 constructor(registeredSignals,statePath=null) {
  this.registered=Object.freeze([...registeredSignals]);
  this.path=statePath;this.used=readState(statePath);
  this.key=registeredSignals.slice(-2).join(':');
  this.pending=false;
 }
 async accept(signals, verify) {
  if(this.path)this.used=readState(this.path);
  if(this.pending||this.used.has(this.key)||signals.length!==this.registered.length) return false;
  const matches=signals.every((s,i)=>i===1?(s==='0'||s==='1'):s===this.registered[i]);
  if(!matches)return false;
  this.pending=true;
  try {
   if(!await verify(signals))return false;
   if(this.path)this.used=readState(this.path);
   if(this.used.has(this.key))return false;
   this.used.add(this.key);saveState(this.path,this.used);return true;
  } finally {this.pending=false}
 }
}
// Enrollment and the approved model/cutoff list are trusted local configuration.
// Coordinator requests cannot create a new batch ID or replace this configuration.
class ClientQueryPolicy {
 constructor(enrollment,statePath=null) {
  if(!enrollment || typeof enrollment.batch_id!=='string' || !enrollment.batch_id ||
     typeof enrollment.root!=='string' || !enrollment.root ||
     !Array.isArray(enrollment.approved_requests) || !enrollment.approved_requests.length)
   throw Error('Authenticated enrollment configuration required');
  this.batchId=enrollment.batch_id;this.root=enrollment.root;
  this.approved=new Set(enrollment.approved_requests.map(r=>{
   if(typeof r.model_digest!=='string'||!r.model_digest||typeof r.threshold!=='string'||!/^\d+$/.test(r.threshold))
    throw Error('Invalid approved request');
   return JSON.stringify([r.model_digest,r.threshold]);
  }));
  this.path=statePath;
  this.used=new Set();
  this.refresh();
 }
 refresh(){
  const statePath=this.path;
  if(statePath&&fs.existsSync(statePath)) {
   const state=JSON.parse(fs.readFileSync(statePath));
   // Old per-model ledgers cannot safely be treated as an unused batch ledger.
   if(state.version!==2||!Array.isArray(state.consumed_batches)||
      !state.consumed_batches.every(x=>typeof x==='string'))
    throw Error('Unsupported query ledger; explicit enrollment migration required');
   this.used=new Set(state.consumed_batches);
  }
 }
 authorize(input) {
  let digest;try{digest=modelDigest(input.weights,input.bias)}catch{return false}
  const root=input.dataset_root,threshold=input.threshold;
  this.refresh();
  if(root!==this.root || !this.approved.has(JSON.stringify([digest,threshold])) ||
     this.used.has(this.batchId))return false;
  // Consume before proving, including failed/aborted attempts. Fail closed on I/O errors.
  this.used.add(this.batchId);
  if(this.path) {
   fs.writeFileSync(this.path+'.tmp',JSON.stringify({version:2,consumed_batches:[...this.used]}));
   fs.renameSync(this.path+'.tmp',this.path);
  }
  return true;
 }
}
// Only call recordVerified after proof verification and registered-request matching.
// requestId identifies the authenticated model/cutoff/round/cohort/deadline schedule.
// The cohort and required passing count are frozen before response collection.
class CohortReleasePolicy {
 constructor(requestId,batchIds,requiredPasses,statePath=null) {
  if(typeof requestId!=='string'||!requestId||!Array.isArray(batchIds)||!batchIds.length||
     !batchIds.every(id=>typeof id==='string'&&id.length)||
     new Set(batchIds).size!==batchIds.length||
     !Number.isSafeInteger(requiredPasses)||requiredPasses<1||requiredPasses>batchIds.length)
   throw Error('Nonempty unique cohort and valid required passing count required');
  this.requestId=requestId;this.ids=Object.freeze([...batchIds]);this.members=new Set(this.ids);
  this.requiredPasses=requiredPasses;this.path=statePath;
  this.results=new Map();this.finalized=false;this.refresh();
 }
 refresh() {
  if(!this.path||!fs.existsSync(this.path))return;
  const state=JSON.parse(fs.readFileSync(this.path));
  if(state.version!==1||state.request_id!==this.requestId||JSON.stringify(state.batch_ids)!==JSON.stringify(this.ids)||
     state.required_passes!==this.requiredPasses||typeof state.finalized!=='boolean'||
     !Array.isArray(state.results)||state.results.some(r=>!Array.isArray(r)||r.length!==2||
       !this.members.has(r[0])||(r[1]!=='0'&&r[1]!=='1'))||
     new Set(state.results.map(r=>r[0])).size!==state.results.length)
   throw Error('Cohort ledger does not match the frozen release request');
  this.results=new Map(state.results);this.finalized=state.finalized;
 }
 persist() {
  if(!this.path)return;
  fs.writeFileSync(this.path+'.tmp',JSON.stringify({version:1,request_id:this.requestId,batch_ids:this.ids,
   required_passes:this.requiredPasses,results:[...this.results],finalized:this.finalized}));
  fs.renameSync(this.path+'.tmp',this.path);
 }
 recordVerified(batchId,bit) {
  this.refresh();
  if(this.finalized||!this.members.has(batchId)||this.results.has(batchId)||
     (bit!=='0'&&bit!=='1'))return false;
  this.results.set(batchId,bit);this.persist();return true;
 }
 summary() {
  this.refresh();
  const N=this.ids.length,p=[...this.results.values()].filter(bit=>bit==='1').length;
  const f=this.results.size-p,u=N-p-f;
  return {request_id:this.requestId,registered:N,required_passes:this.requiredPasses,positive:p,negative:f,unresolved:u,
   respondent_positive_fraction:p+f?p/(p+f):null,coverage:(p+f)/N,
   lower_bound:p/N,upper_bound:(p+u)/N,finalized:this.finalized,
   decision:p>=this.requiredPasses?'certified_release':
    p+u<this.requiredPasses?'certified_reject':'inconclusive'};
 }
 finalize() {this.refresh();this.finalized=true;this.persist();return this.summary()}
}
module.exports={VerifierPolicy,ClientQueryPolicy,CohortReleasePolicy,modelDigest};
