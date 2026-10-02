"""Reproducible committed-batch evaluation of quantized ridge classifiers.

Training, PCA fitting and batch provisioning are separate from proof timings.
This replaces unsupported end-to-end CNN claims with an actual in-circuit model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import secrets
import subprocess
import sys
import time
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_loader import load_mnist, download_and_extract_har, load_har

D, B, SCALE = 16, 8, 100

def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def node(req,path):
    request=path.with_suffix('.request.json')
    write(request,req)
    subprocess.run(['node','scripts/zk_engine.cjs',str(request),str(path)],check=True)
    request.unlink()
    return json.loads(path.read_text())

def loss_sum(x,y,w,b):
    scores=x.astype(np.int64)@w.astype(np.int64).T+b
    return int(np.square(scores-y*10000).sum())

def run(args):
    os.chdir(ROOT)
    out=ROOT/'results'/'committed-rerun'
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    manifest={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
      'processor':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
      'torch':torch.__version__,'numpy':np.__version__,
      'configuration':vars(args),'batch_size':B,'dimensions':D,'scale':SCALE,
      'training':'ridge alpha=1.0; PCA train-only; 30% official training split',
      'provisioning':'simulated trusted registry; disjoint ordered test batches per configuration',
      'privacy':'local losses retained only for research analysis; not protocol messages'}
    write(out/'manifest.json',manifest)
    rows=[];tasks=[];quality=[];security=None
    for dataset in args.datasets:
        if dataset=='mnist':
            train,test=load_mnist('data')
            xtrain=train.data.numpy().reshape(-1,784)/255.
            xtest=test.data.numpy().reshape(-1,784)/255.
            ytrain=train.targets.numpy();ytest=test.targets.numpy();classes=10
        else:
            download_and_extract_har('https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip','data')
            train,test=load_har('data')
            xtrain=train.features.numpy();xtest=test.features.numpy()
            ytrain=train.labels.numpy();ytest=test.labels.numpy();classes=6
        # Shared ten-output circuit; HAR pads four unused target/model outputs with zero.
        for seed in args.seeds:
            rng=np.random.default_rng(seed)
            training_ids=rng.choice(len(xtrain),int(.3*len(xtrain)),replace=False)
            t=time.perf_counter()
            pca=PCA(n_components=D,svd_solver='randomized',random_state=seed)
            ztrain=pca.fit_transform(xtrain[training_ids]);ztest=pca.transform(xtest)
            model=Ridge(alpha=1.0).fit(ztrain,np.eye(10)[ytrain[training_ids]])
            w=np.rint(model.coef_*SCALE).astype(np.int64)
            b=np.rint(model.intercept_*SCALE*SCALE).astype(np.int64)
            x=np.rint(ztest*SCALE).astype(np.int64)
            assert np.abs(w).max()<32768 and np.abs(x).max()<32768
            assert np.abs(b).max()<34359738368
            train_s=time.perf_counter()-t
            full_scores=x@w.T+b
            float_loss=np.mean((model.predict(ztest)-np.eye(10)[ytest])**2)
            fixed_loss=np.mean((full_scores/10000-np.eye(10)[ytest])**2)
            model_hash=hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()
            quality.append({'dataset':dataset,'seed':seed,'train_s':train_s,
              'test_accuracy':float(np.mean(full_scores[:,:classes].argmax(1)==ytest)),
              'test_mse':float(fixed_loss),'float_mse':float(float_loss),
              'mse_quantization_error':float(abs(fixed_loss-float_loss)),
              'train_samples':len(training_ids),'test_samples':len(ytest),
              'model_sha256':model_hash})
            for non_iid in args.distributions:
                order=rng.permutation(len(ytest))
                if non_iid=='label_sorted': order=order[np.argsort(ytest[order],kind='stable')]
                for n in args.clients:
                    batches=[];batch_indices=[]
                    # Evenly spaced test subsets; each client gets exactly B records.
                    chunks=np.array_split(order,n)
                    for i,chunk in enumerate(chunks):
                        ids=chunk[:B];assert len(ids)==B
                        batch_indices.append(ids)
                        random_salt=secrets.randbits(128)
                        batches.append({'client_id':str(i+1),'salt':str(random_salt),
                           'features':x[ids].astype(str).tolist(),
                           'labels':np.eye(10,dtype=np.int64)[ytest[ids]].astype(str).tolist()})
                    roots=node({'op':'enroll','items':batches},out/'enrollment.json')
                    for i,(item,root,ids) in enumerate(zip(batches,roots,batch_indices)):
                        t=time.perf_counter()
                        ls=loss_sum(x[ids],np.eye(10,dtype=np.int64)[ytest[ids]],w,b)
                        baseline_s=time.perf_counter()-t
                        timings=[]
                        for _ in range(25):
                            t=time.perf_counter()
                            loss_sum(x[ids],np.eye(10,dtype=np.int64)[ytest[ids]],w,b)
                            timings.append(time.perf_counter()-t)
                        baseline_s=float(np.median(timings))
                        local_mse=ls/(B*10*100000000)
                        accuracy=float(np.mean(full_scores[ids,:classes].argmax(1)==ytest[ids]))
                        for threshold in args.thresholds:
                            cutoff=int(np.floor(threshold*B*10*100000000))
                            passed=ls<cutoff
                            meta={'dataset':dataset,'seed':seed,'distribution':non_iid,
                              'num_clients':n,'client_id':i+1,'threshold':threshold,
                              'local_mse':local_mse,'local_accuracy':accuracy,
                              'loss_sum':str(ls),'threshold_sum':str(cutoff),
                              'passes':passed,'baseline_eval_s':baseline_s,
                              'dataset_root':root,'model_sha256':model_hash}
                            rows.append(meta)
                            inp={**item,'dataset_root':root,'weights':w.astype(str).tolist(),
                              'bias':b.astype(str).tolist(),'threshold':str(cutoff),'nonce':str(seed*100000+n)}
                            tasks.append({'meta':meta,'input':inp})
                            if security is None and ls>0:
                                security={'input':{**inp,'threshold':str(ls+1)},'loss_sum':str(ls)}
            print(f'Prepared {dataset} seed {seed}: accuracy={quality[-1]["test_accuracy"]:.3f}',flush=True)
            write(out/'local_outcomes.json',rows);write(out/'model_quality.json',quality)
    paths={'wasm':'zkp_setup/committed/committed_linear_js/committed_linear.wasm',
           'zkey':'zkp_setup/committed/committed_linear_final.zkey',
           'vkey':'zkp_setup/committed/verification_key.json'}
    if not args.prepare_only:
        node({'op':'bench',**paths,'tasks':tasks},out/'proof_measurements.json')
        node({'op':'security',**paths,**security,'verify_counts':args.verify_counts},out/'security.json')
    manifest['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    manifest['outcomes']=len(rows);manifest['proof_tasks']=len(tasks)
    write(out/'manifest.json',manifest)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',default=['mnist','har'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--clients',nargs='+',type=int,default=[5,10])
    p.add_argument('--thresholds',nargs='+',type=float,default=[.025,.05,.1])
    p.add_argument('--distributions',nargs='+',default=['iid','label_sorted'])
    p.add_argument('--verify-counts',nargs='+',type=int,default=[50,100,500,1000])
    p.add_argument('--prepare-only',action='store_true')
    run(p.parse_args())
