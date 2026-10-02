"""Independent validation-threshold/model-quality study; retain proof provenance."""
import hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import scipy,sklearn,torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from data_loader import load_mnist,load_har
from scripts.rerun_committed import node,loss_sum
OUT=ROOT/'results/utility-study';OUT.mkdir(parents=True,exist_ok=True)
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False))
paths={'wasm':'zkp_setup/committed/committed_linear_js/committed_linear.wasm','zkey':'zkp_setup/committed/committed_linear_final.zkey','vkey':'zkp_setup/committed/verification_key.json'}
def main():
 torch.set_num_threads(4);models=[];tasks=[];curves=[];security=None
 for dataset in ['mnist','har']:
  train,test=load_mnist('data') if dataset=='mnist' else load_har('data')
  if dataset=='mnist':
   xt=train.data.numpy().reshape(-1,784)/255.;xe=test.data.numpy().reshape(-1,784)/255.
   yt=train.targets.numpy();ye=test.targets.numpy();classes=10
  else:xt=train.features.numpy();xe=test.features.numpy();yt=train.labels.numpy();ye=test.labels.numpy();classes=6
  split=np.random.default_rng(20261001).permutation(len(yt));nv=int(.1*len(yt));val=split[:nv];pool=split[nv:]
  testids=np.random.default_rng(20261002).permutation(len(ye))[:400].reshape(50,8)
  validids=val[:min(100,len(val)//8)*8].reshape(-1,8)
  references={};cutoff=None
  # Fit the reference first so the validation-selected cutoff is frozen before test comparisons.
  for seed,fraction in [(42,.3)]+[(s,f) for s in [42,43,44] for f in [.01,.05,.3] if (s,f)!=(42,.3)]:
   t=time.perf_counter();ids=np.random.default_rng(seed).permutation(pool)[:int(fraction*len(pool))]
   pca=PCA(n_components=16,svd_solver='randomized',random_state=seed)
   z=pca.fit_transform(xt[ids]);model=Ridge(alpha=1).fit(z,np.eye(10)[yt[ids]])
   w=np.rint(model.coef_*100).astype(np.int64);b=np.rint(model.intercept_*10000).astype(np.int64)
   x=np.rint(pca.transform(xe)*100).astype(np.int64)
   assert np.abs(x).max()<32768 and np.abs(w).max()<32768 and np.abs(b).max()<2**35
   if cutoff is None:
    xv=np.rint(pca.transform(xt[val])*100).astype(np.int64)
    vloss=[loss_sum(xv[i:i+8],np.eye(10,dtype=np.int64)[yt[val[i:i+8]]],w,b) for i in range(0,len(validids)*8,8)]
    cutoff=int(np.floor(np.median(vloss)))
    references={'dataset':dataset,'reference_seed':42,'reference_fraction':.3,'validation_batches':len(vloss),'threshold_sum':str(cutoff),'threshold':cutoff/8e9,'validation_loss_sums':[str(v) for v in vloss],'method':'median integer loss, floor; disjoint official-training validation split'}
    save(dataset+'-threshold.json',references)
   scores=x@w.T+b;accuracy=float(np.mean(scores[:,:classes].argmax(1)==ye))
   losses=[loss_sum(x[ix],np.eye(10,dtype=np.int64)[ye[ix]],w,b) for ix in testids]
   acc=[float(np.mean(scores[ix,:classes].argmax(1)==ye[ix])) for ix in testids]
   digest=hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()
   entry={'dataset':dataset,'seed':seed,'fraction':fraction,'train_samples':len(ids),'full_test_accuracy':accuracy,
    'pass_fraction':float(np.mean(np.array(losses)<cutoff)),'threshold_sum':str(cutoff),'threshold':cutoff/8e9,
    'model_sha256':digest,'batch_loss_sums':[str(v) for v in losses],'batch_accuracies':acc,'test_indices':testids.tolist(),'fit_and_evaluate_s':time.perf_counter()-t}
   models.append(entry)
   # One genuine circuit proof per model; the full 50-batch CDF remains a cleartext analysis.
   inp={'client_id':'1','salt':str(int.from_bytes(os.urandom(16),'big')),
    'features':x[testids[0]].astype(str).tolist(),'labels':np.eye(10,dtype=np.int64)[ye[testids[0]]].astype(str).tolist(),
    'weights':w.astype(str).tolist(),'bias':b.astype(str).tolist(),'threshold':str(cutoff),'nonce':str(seed*100000+int(fraction*100))}
   inp['dataset_root']=node({'op':'enroll','items':[inp]},OUT/'enroll.json')[0]
   tasks.append({'meta':{'dataset':dataset,'seed':seed,'training_fraction':fraction,'threshold':cutoff/8e9,'num_clients':1,'client_id':1,'reference_loss_sum':str(losses[0])},'input':inp})
   if security is None:security={'input':{**inp,'threshold':str(losses[0]+1)},'loss_sum':str(losses[0])}
   print(f'{dataset} seed={seed} fraction={fraction}: accuracy={accuracy:.4f}, pass={entry["pass_fraction"]:.2f}',flush=True)
   save('models.json',models)
 node({'op':'bench',**paths,'tasks':tasks},OUT/'proof_measurements.json')
 node({'op':'security',**paths,**security,'verify_counts':[]},OUT/'security-checks.json')
 relations=[]
 for ds in ['mnist','har']:
  a=[m for m in models if m['dataset']==ds];r=spearmanr([m['full_test_accuracy'] for m in a],[m['pass_fraction'] for m in a])
  relations.append({'dataset':ds,'models':len(a),'spearman_accuracy_pass':float(r.statistic) if np.isfinite(r.statistic) else None,
   'interpretation':'descriptive matched-batch comparison; correlated models; no independent-model significance claim'})
 save('summary.json',{'models':models,'relations':relations,'proof_count':len(tasks),'analysis_batch_count':len(models)*50,
  'selection':'10% official training reserved for validation; fractions of remaining 90%; identical 50 disjoint test batches per dataset across models',
  'software':{'numpy':np.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'torch':torch.__version__},
  'hardware':{'physical_cores':int(subprocess.check_output(['sysctl','-n','hw.physicalcpu'])),'logical_cores':int(subprocess.check_output(['sysctl','-n','hw.logicalcpu'])),'ram_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize']))},
  'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
if __name__=='__main__':main()
