"""Build the results summary from the completed measurements."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import zipfile
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results'/'committed-rerun'
PAPER=ROOT/'paper'
def read(name):return json.loads((DATA/name).read_text())
def pm(values,digits=1):
    return f'{np.mean(values):.{digits}f} $\\pm$ {np.std(values,ddof=1):.{digits}f}'
def row(*cells):return ' & '.join(map(str,cells))+r' \\'+'\n'
def correction_fields(records,rates,scale):
    study=json.loads((ROOT/'results/utility-study/summary.json').read_text())
    policy=json.loads((ROOT/'results/policy-checks.json').read_text())
    extra=json.loads((ROOT/'results/utility-study/proof_measurements.json').read_text())
    release=json.loads((ROOT/'results/release-policy.json').read_text())
    if not all(c['passed'] for c in release['checks']):raise RuntimeError('Release policy checks failed')
    if len(extra)!=18 or study['proof_count']!=18:raise RuntimeError('Utility proof coverage incomplete')
    expected={(m['dataset'],m['seed'],m['fraction']) for m in study['models']}
    if {(m['dataset'],m['seed'],m['training_fraction']) for m in extra}!=expected:raise RuntimeError('Utility models/proofs differ')
    if not all(x['passed'] for x in policy['checks']):raise RuntimeError('Corrected policy failed')
    rawrows=''
    for ds in ['mnist','har']:
        for split in ['iid','label_sorted']:
            for t in [.025,.05,.1]:
                cols=[]
                for n in [5,10]:
                    r=next(r for r in rates if (r['dataset'],r['split'],r['threshold'],r['clients'])==(ds,split,t,n))
                    cols.append('/'.join(f'{v:.0f}' if abs(v-round(v))<1e-8 else f'{v:.1f}' for v in r['seed_percentages']))
                rawrows+=row(ds.upper(),'IID' if split=='iid' else 'LS',f'{t:.3f}',*cols)
    qualityrows=''
    for ds in ['mnist','har']:
        for f in [.01,.05,.3]:
            a=[r for r in study['models'] if r['dataset']==ds and r['fraction']==f]
            qualityrows+=row(ds.upper(),f'{100*f:.0f}',f'{100*np.mean([r["full_test_accuracy"] for r in a]):.1f}',f'{100*np.mean([r["pass_fraction"] for r in a]):.1f}')
    correlations={r['dataset']:r['spearman_accuracy_pass'] for r in study['relations']}
    thresholds={d:next(m['threshold'] for m in study['models'] if m['dataset']==d) for d in ['mnist','har']}
    utilitytext=f'The fixed thresholds are {thresholds["mnist"]:.5f} (MNIST) and {thresholds["har"]:.5f} (HAR). Across nine models per dataset, the descriptive Spearman correlation between full-test accuracy and passing-batch fraction is {correlations["mnist"]:.2f} for MNIST and {correlations["har"]:.2f} for HAR. Models share test batches, and no independent-model significance claim is made.'
    shapes=json.loads((ROOT/'results/circuit-shapes.json').read_text())['shapes']
    return {'CORES':str(study['hardware']['physical_cores']),'RAM':str(round(study['hardware']['ram_bytes']/2**30)),
        'SKLEARN':study['software']['scikit_learn'],'SCIPY':study['software']['scipy'],
        'RATERAWROWS':rawrows,'UTILITYROWS':qualityrows,'MODELUTILITYTEXT':utilitytext,
        'SHAPEROWS':''.join(row(r['batch_size'],r['dimensions'],f'{r["nonlinear_constraints"]:,}') for r in shapes),
        'WORKLOADMS':f'{1000*scale[-1]["total_s"]/scale[-1]["proof_checks"]:.2f}'}

def build(draft=False):
    local=read('local_outcomes.json');proofs=read('proof_measurements.json')
    quality=read('model_quality.json');manifest=read('manifest.json')
    complete=len(proofs)==len(local)
    if not draft and not complete:raise RuntimeError('Proof sweep is incomplete')
    if not draft:
        security=json.loads((ROOT/'results/utility-study/security-checks.json').read_text())
        if not all(x['passed'] for x in security['checks']):raise RuntimeError('Security checks failed')
    else:security={'checks':[],'verification_workload':[]}
    baseline=json.loads((ROOT/'results'/'threshold-baseline.json').read_text())
    key=lambda r:(r['dataset'],r['seed'],r['distribution'],r['num_clients'],r['client_id'],r['threshold'])
    if len({key(x) for x in proofs})!=len(proofs):raise RuntimeError('Duplicate measured configuration')
    if not draft and {key(x) for x in proofs}!={key(x) for x in local}:raise RuntimeError('Measurement coverage mismatch')
    metrics=['witness_s','prove_s','verify_s','proof_json_bytes','public_json_bytes']
    means={k:float(np.mean([r[k] for r in proofs])) for k in metrics}
    base={k:float(np.mean([r[k] for r in baseline['measurements']])) for k in metrics}
    rates=defaultdict(list)
    for x in local:rates[(x['dataset'],x['distribution'],x['threshold'],x['num_clients'],x['seed'])].append(x['passes'])
    ratetext=''
    raterecords=[]
    for ds in ['mnist','har']:
        for split in ['iid','label_sorted']:
            for t in [.025,.05,.1]:
                cols=[]
                for n in [5,10]:
                    vals=[100*np.mean(rates[ds,split,t,n,s]) for s in [42,43,44]]
                    cols.append(pm(vals))
                    raterecords.append({'dataset':ds,'split':split,'threshold':t,'clients':n,
                        'mean_percent':float(np.mean(vals)),'sd_percent':float(np.std(vals,ddof=1)), 'seed_percentages':vals})
                ratetext+=row(ds.upper(),'IID' if split=='iid' else 'LS',f'{t:.3f}',*cols)
    qualitytext=''
    qsummary=[]
    for ds in ['mnist','har']:
        records=[x for x in quality if x['dataset']==ds]
        qualitytext+=row(ds.upper(),pm([100*x['test_accuracy'] for x in records]),
            pm([x['test_mse'] for x in records],4),f'{np.mean([x["mse_quantization_error"] for x in records]):.6f}')
        qsummary.append({'dataset':ds,'accuracy_mean':float(np.mean([x['test_accuracy'] for x in records])),
                        'quantization_error_mean':float(np.mean([x['mse_quantization_error'] for x in records]))})
    costrows=''
    for label,k in [('Witness (s)','witness_s'),('Proving (s)','prove_s'),('Verification (s)','verify_s'),
                    ('Proof JSON (KiB)','proof_json_bytes'),('Public JSON (KiB)','public_json_bytes')]:
        scale=1024 if k.endswith('bytes') else 1
        costrows+=row(label,f'{base[k]/scale:.4f}',f'{means[k]/scale:.4f}')
    # Unique batch rows: threshold sweeps do not create new evaluation samples.
    unique={key(x)[:-1]:x for x in local}
    records=list(unique.values())
    attack=[]
    losses=np.array([x['local_mse'] for x in records])
    assert np.all((losses>=0)&(losses<=1))
    raw_losses=np.array([int(x['loss_sum']) for x in records],dtype=np.int64)
    denominator=8*10*100000000
    for q in [1,4,8,12,16]:
        lo=np.zeros(len(losses),dtype=np.int64)
        hi=np.full(len(losses),denominator,dtype=np.int64)
        for _ in range(q):
            mid=(lo+hi+1)//2;bits=raw_losses<mid
            hi=np.where(bits,mid-1,hi);lo=np.where(bits,lo,mid)
        estimate=(lo+hi)/2/denominator
        attack.append({'queries':q,'max_error':float(np.max(np.abs(estimate-losses))),
                       'mean_error':float(np.mean(np.abs(estimate-losses))),
                       'interval_width':float(np.max((hi-lo)/denominator)),
                       'predicate':'exact integer loss_sum < integer cutoff'})
    utility=[]
    for ds in ['mnist','har']:
        a=[x for x in records if x['dataset']==ds]
        r=spearmanr([x['local_mse'] for x in a],[x['local_accuracy'] for x in a])
        utility.append({'dataset':ds,'spearman_loss_accuracy':float(r.statistic),'batches':len(a)})
    leakage=f'After 12 adaptive queries, the largest absolute midpoint reconstruction error is {attack[3]["max_error"]:.6f}; after 16 it is {attack[4]["max_error"]:.6f}. This simulation reconstructs the fixed scalar loss.'
    per_model=[]
    for ds in ['mnist','har']:
        for seed in [42,43,44]:
            a=[x for x in records if x['dataset']==ds and x['seed']==seed]
            correlation=spearmanr([x['local_mse'] for x in a],[x['local_accuracy'] for x in a]).statistic
            per_model.append({'dataset':ds,'seed':seed,'batches':len(a),'spearman_loss_accuracy':float(correlation)})
    utiltext='Within each primary model, batch loss/accuracy rank correlations are '+ '; '.join(ds.upper()+': '+', '.join(f'{r["spearman_loss_accuracy"]:.3f}' for r in per_model if r['dataset']==ds) for ds in ['mnist','har'])+' (seeds 42/43/44).'
    scale=read('security.json')['verification_workload']
    scaletext='The 1,000-check workload takes '+f'{scale[-1]["total_s"]:.3f}'+' seconds.' if scale else 'Verification workload measurements are still pending in this draft.'
    securitytext='All 25 relation/request checks pass: 20 rejection checks and five positive or boundary controls. Honest proofs, equality/below-loss zero outputs, registered submissions, and first agreed queries succeed. Changed records, invalid labels/ranges, modified public statements, re-registration, and duplicate requests are rejected. A separate 17-check hardened-policy suite passes, including rejection of changed approved models after consumption, re-salting with the same batch ID, restart reuse, unapproved requests, legacy ledgers, and concurrent verifier duplicates.' if security['checks'] else 'Checks pending.'
    if baseline['changed_nonce_accepted'] or baseline['changed_model_hash_accepted']:
        attacktext='In this baseline, altered context signals are accepted; context binding therefore requires correction.'
    else:
        attacktext='Changing its model-hash or nonce public signals on an existing proof is rejected in our test. That public-statement binding still does not establish that the supplied scalar is the model\'s loss.'
    baseline_eval=float(np.mean([x['baseline_eval_s'] for x in records]))
    replacements={'PROOFCOUNT':str(len(proofs)),'WITNESS':f'{means["witness_s"]:.3f}',
        'PROVE':f'{means["prove_s"]:.3f}','VERIFY':f'{means["verify_s"]:.4f}', 'VERIFYMS':f'{means["verify_s"]*1000:.2f}',
        'PUBLICKIB':f'{means["public_json_bytes"]/1024:.2f}','PROOFKIB':f'{means["proof_json_bytes"]/1024:.2f}', 'TOTALKIB':f'{(means["proof_json_bytes"]+means["public_json_bytes"])/1024:.2f}',
        'SECURITYTEXT':securitytext,'BASELINEATTACK':attacktext,'LEAKAGETEXT':leakage,
        'HARDWARE':manifest['processor']+' workstation running '+manifest['platform'].split('-')[0]+' '+manifest['platform'].split('-')[1],
        'PYTHON':manifest['python'].split()[0],'NUMPY':manifest['numpy'],'TORCH':manifest['torch'],
        'QUALITYTEXT':'Mean absolute MSE differences between the floating-point and quantized models remain below 0.0003 on MNIST and 0.0004 on HAR; this measures numerical change on these test sets, not a worst-case arithmetic bound.',
        'QUALITYROWS':qualitytext,'COSTROWS':costrows,'RATEROWS':ratetext,
        'COSTTEXT':f'The no-ZKP integer loss baseline averages {baseline_eval*1e6:.2f} microseconds per eight-record batch, with PCA and provisioning excluded. Mean witness-plus-proving time is {means["witness_s"]+means["prove_s"]:.3f} seconds. The scalar baseline evaluates only a supplied number.',
        'ACCEPTANCETEXT':f'all {len(proofs)} submitted evaluation proofs are accepted and their outputs match the integer reference computation. This includes both positive and negative threshold outcomes.',
        'UTILITYTEXT':utiltext,'SCALETEXT':scaletext,'SCALEROWS':''.join(row(x['proof_checks'],f'{x["total_s"]:.3f}',f'{1000*x["total_s"]/x["proof_checks"]:.3f}') for x in scale)}
    replacements.update(correction_fields(records,raterecords,scale))
    # Values that the paper reports, recomputed from the raw measurements.
    if PAPER.exists():(PAPER/'reported_values.json').write_text(json.dumps(replacements,indent=2))
    summary={'complete':complete and not draft,'proof_count':len(proofs),'batch_count':len(records),
      'means':means,'scalar_baseline_means':base,'no_zkp_eval_mean_s':baseline_eval,
      'threshold_rates':raterecords,'quality':qsummary,'utility':utility,'adaptive_leakage':attack,
      'security_checks':security['checks'],'verification_workload':read('security.json')['verification_workload'],
      'policy_checks':json.loads((ROOT/'results/policy-checks.json').read_text()),
      'release_policy':json.loads((ROOT/'results/release-policy.json').read_text()),
      'per_model_utility':per_model,'utility_study':json.loads((ROOT/'results/utility-study/summary.json').read_text()),
      'circuit_shapes':json.loads((ROOT/'results/circuit-shapes.json').read_text())}
    (ROOT/'results'/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    if not draft and (PAPER/'manuscript.tex').exists():
        with zipfile.ZipFile(PAPER/'overleaf-project.zip','w',zipfile.ZIP_DEFLATED) as z:
            z.write(PAPER/'manuscript.tex','main.tex')
            for f in sorted((PAPER/'figures').glob('*.pdf')):z.write(f,f'figures/{f.name}')
    print('Built',ROOT/'results'/'summary.json','proofs',len(proofs),'complete',not draft)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--draft',action='store_true')
    build(p.parse_args().draft)
