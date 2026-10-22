"""Offline, synthetic end-to-end checks of the custom trainer + new-patient scoring.
Run with: PYTHONPATH=src OMP_NUM_THREADS=1 python test_end_to_end.py
Does not measure clinical predictiveness; only interface, integrity and calculation correctness.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
import torch
import subprocess
import sys
import os
from nsclc_regimen_mil.custom import create_model, predict_custom, H5ADCells, _embedding, _normalize, _seed
from nsclc_regimen_mil.model import TreatmentConditionedMultiHeadAttentionMIL

BASE_SEED=918
RNG=np.random.default_rng(BASE_SEED)
GENES=[f"GENE{i:03d}" for i in range(64)]

def save_h5ad(path, values, genes, obs=None, *, dense=False, categorical=None):
    m=sp.csr_matrix(values.astype(np.float32))
    with h5py.File(path,'w') as h:
        if dense:
            h.create_dataset('X', data=m.toarray())
            lay=h.create_group('layers');lay.create_dataset('counts',data=m.toarray())
        else:
            g=h.create_group('X');g.attrs['shape']=m.shape;g.attrs['encoding-type']='csr_matrix'
            for name,data in {'indptr':m.indptr,'indices':m.indices,'data':m.data}.items():g.create_dataset(name,data=data)
            lay=h.create_group('layers');lay['counts']=g
        var=h.create_group('var');var.attrs['_index']='gene_id'
        var.create_dataset('gene_id',data=np.asarray(genes,dtype='S'))
        obs_group=h.create_group('obs')
        for name,v in (obs or {}).items():
            if categorical==name:
                choices=sorted(set(v));g=obs_group.create_group(name)
                g.create_dataset('categories',data=np.asarray(choices,dtype='S'))
                g.create_dataset('codes',data=np.asarray([choices.index(x) for x in v],dtype=np.int16))
            else:
                obs_group.create_dataset(name,data=np.asarray(v,dtype='S'))


def must_fail(fn, fragment):
    try:fn()
    except Exception as e:
        if fragment.lower() not in str(e).lower():
            raise AssertionError(f"Expected {fragment!r}, got {type(e).__name__}: {e}") from e
        return
    raise AssertionError(f"Expected failure containing {fragment!r}, but succeeded")


def make_patient_rows(pid, regimen, label, n_cells):
    rows=RNG.poisson(0.6,size=(n_cells,len(GENES))).astype(np.float32)
    rows[:,3]+=(label=="R")*5
    rows[:,6]+=(regimen=="DrugB+Chemo")*4
    return rows

def main(tmp):
    root=Path(tmp)
    data=[]; patients=[]; treatments=[]; responses=[]
    for ti,reg in enumerate(('DrugA','DrugB+Chemo')):
        for ri,resp in enumerate(('R','NR')):
            for ix in range(3):
                patient=f"TRAIN_{ti}_{ri}_{ix}"
                n=15+ix
                data.append(make_patient_rows(patient,reg,resp,n)); patients.extend([patient]*n)
                treatments.extend([reg]*n);responses.extend([resp]*n)
    train_m=np.vstack(data)
    save_h5ad(root/'train.h5ad',train_m,GENES,
              {'Patient':patients,'Treatment':treatments,'response_binary':responses},categorical='Patient')
    model_dir=root/'model'
    create_model(root/'train.h5ad',model_dir,epochs=2,cells_per_bag=8,pool_cells=16,seed=123,device='cpu')
    assert all((model_dir/f).exists() for f in ('custom_mil.pt','gene_order.csv',
        'treatment_vocabulary.csv','reference_centroids.npz','reference_coverage.csv','training_metadata.json'))
    print('PASS 1: create_model generated full model bundle and same-model reference bank')

    # Novel patients without regimen or response: scoring needs only the count matrix and optional patient IDs.
    queries=[];qid=[]
    for p,n,reg,lab in [('NEW_A',23,'DrugA','R'),('NEW_B',20,'DrugB+Chemo','NR')]:
        queries.append(make_patient_rows(p,reg,lab,n));qid.extend([p]*n)
    query_m=np.vstack(queries)
    save_h5ad(root/'novel.h5ad',query_m,GENES,{'Patient':qid},categorical='Patient')
    def score(file,output,name='Patient',pid=None, min_cov=0.8):
        return predict_custom(file,model_dir,output,patient_col=name,patient_id=pid,
                              layer='counts',max_cells=10,repeats=3,min_gene_coverage=min_cov,seed=72,device='cpu')
    standard=score(root/'novel.h5ad',root/'scores.csv')
    raw=pd.read_csv(root/'scores_repeats.csv')
    assert len(standard)==4 and len(raw)==12
    assert set(standard.Patient)=={'NEW_A','NEW_B'}
    assert standard.groupby('Patient').Exact_rank.apply(lambda r: sorted(r.tolist())==[1,2]).all()
    assert np.allclose(standard.Exact_RminusNR,standard.Mean_R_similarity-standard.Mean_NR_similarity,atol=1e-7)
    assert standard.Exact_RminusNR.between(-2,2).all()
    # Ensure summary matches its 3-repeat mean and sample SD exactly.
    agg=raw.groupby(['Patient','Regimen']).Exact_RminusNR.agg(['mean','std'])
    for _,rr in standard.iterrows():
        item=agg.loc[(rr.Patient,rr.Regimen)]
        assert abs(rr.Exact_RminusNR-item['mean'])<1e-9
        assert abs(rr.Score_SD-item['std'])<1e-9
    print('PASS 2: two genuinely unseen patients, no treatment/response labels, 3 repeats, ranks, CSVs, Exact delta and SD')

    # Test automatic gene alignment with full exact permutation (query gene columns in reverse order).
    rev=np.arange(len(GENES))[::-1]
    save_h5ad(root/'shuffled.h5ad',query_m[:,rev],np.array(GENES)[rev],{'Patient':qid})
    shuffled=score(root/'shuffled.h5ad',root/'shuffled.csv')
    a=standard.sort_values(['Patient','Regimen']).reset_index(drop=True)
    b=shuffled.sort_values(['Patient','Regimen']).reset_index(drop=True)
    assert np.allclose(a.Exact_RminusNR,b.Exact_RminusNR,atol=1e-7)
    assert np.allclose(a.raw_R_score,b.raw_R_score,atol=1e-7)
    print('PASS 3: arbitrary query gene-column permutation leaves scores unchanged')

    # Missing model gene zero-filled; zero-valued extra query gene accepted.
    save_h5ad(root/'extra.h5ad',np.column_stack([query_m,np.zeros(len(qid))]),GENES+['EXTRA_ZERO'],{'Patient':qid})
    extra=score(root/'extra.h5ad',root/'extra.csv')
    assert np.allclose(a.Exact_RminusNR,extra.sort_values(['Patient','Regimen']).Exact_RminusNR,atol=1e-7)
    keep=np.arange(56)
    save_h5ad(root/'missing.h5ad',query_m[:,keep],np.array(GENES)[keep],{'Patient':qid})
    missing=score(root/'missing.h5ad',root/'missing.csv')
    assert len(missing)==4 and np.isfinite(missing.Exact_RminusNR).all()
    must_fail(lambda:score(root/'missing.h5ad',root/'reject.csv',min_cov=0.90),'Insufficient gene coverage')
    print('PASS 4: extra zero-only genes, missing-model genes, and minimum coverage guard')

    # Single patient H5AD without obs.Patient; no treatment or response labels required.
    save_h5ad(root/'single.h5ad',query_m[:23],GENES,{},dense=True)
    single=score(root/'single.h5ad',root/'single.csv',pid='One_new_patient')
    assert len(single)==2 and set(single.Patient)=={'One_new_patient'}
    print('PASS 5: one new patient, no obs metadata, dense raw counts and --patient-id')

    # Detection of accidental scoring of historical/training patient.
    must_fail(lambda:score(root/'train.h5ad',root/'train_scored.csv'),'training/reference patient IDs')
    print('PASS 6: training/reference patients rejected by default')

    # Corruption/accidental mixture of saved checkpoint and reference bank.
    with np.load(model_dir/'reference_centroids.npz',allow_pickle=False) as bank:
        payload={k:np.array(bank[k]) for k in bank.files}
    payload['checkpoint_sha256']=np.array('wronghash')
    np.savez_compressed(model_dir/'reference_centroids.npz',**payload)
    must_fail(lambda:score(root/'novel.h5ad',root/'mismatch.csv'),'checkpoint mismatch')
    # restore as stored SHA
    payload['checkpoint_sha256']=np.array(hashlib.sha256((model_dir/'custom_mil.pt').read_bytes()).hexdigest())
    np.savez_compressed(model_dir/'reference_centroids.npz',**payload)
    print('PASS 7: mismatched checkpoint/reference bank rejected')

    # Gene-order tampering must be blocked.
    genes_orig=pd.read_csv(model_dir/'gene_order.csv')
    altered=genes_orig.iloc[::-1].reset_index(drop=True)
    altered.to_csv(model_dir/'gene_order.csv',index=False)
    must_fail(lambda:score(root/'novel.h5ad',root/'wrong_order.csv'),'Gene-order file does not match')
    genes_orig.to_csv(model_dir/'gene_order.csv',index=False)
    print('PASS 8: tampered gene order rejected')

    # Compare centroid dot-product delta vs individually normalized reference vectors explicitly.
    ck=torch.load(model_dir/'custom_mil.pt',weights_only=True,map_location='cpu')
    mdl=TreatmentConditionedMultiHeadAttentionMIL(**ck['config']);mdl.load_state_dict(ck['state_dict']);mdl.eval()
    with H5ADCells(root/'novel.h5ad','counts') as qr, H5ADCells(root/'train.h5ad','counts') as tr:
        ytr=tr.obs('response_binary'); ptr=tr.obs('Patient'); ttr=tr.obs('Treatment')
        x=_normalize(qr.rows(np.arange(10)),ck['target_sum'])
        for ti,treat in enumerate(ck['regimens']):
            _,qvec=_embedding(mdl,x,ti,torch.device('cpu'))
            rr=[]; nn=[]
            for p in sorted(set(ptr)):
                inds=np.flatnonzero(ptr==p)
                if ttr[inds[0]]!=treat:continue
                if len(inds)>16:
                    inds = np.sort(np.random.default_rng(_seed(123,p)).choice(inds,16,replace=False))
                xref=_normalize(tr.rows(inds),ck['target_sum'])
                _,vec=_embedding(mdl,xref,ti,torch.device('cpu'))
                (rr if ytr[inds[0]]=='R' else nn).append(vec)
            expected=np.mean([qvec@v for v in rr])-np.mean([qvec@v for v in nn])
            with np.load(model_dir/'reference_centroids.npz',allow_pickle=False) as bank:
                actual=qvec@bank['R'][ti]-qvec@bank['NR'][ti]
            assert abs(expected-actual)<1e-8
    print('PASS 9: Exact R−NR centroid math shape and consistency checked')
    # Missing/contradictory training labels: reject, not silently pool.
    wrong_responses=responses.copy();wrong_responses[0]='NR' if responses[0]=='R' else 'R'
    save_h5ad(root/'contradiction.h5ad',train_m,GENES,
              {'Patient':patients,'Treatment':treatments,'response_binary':wrong_responses})
    must_fail(lambda:create_model(root/'contradiction.h5ad',root/'invalid',epochs=1,device='cpu'),'conflicting treatment/response')
    # noninteger values rejected
    non_integer=query_m.copy();non_integer[:,0]=1.5
    save_h5ad(root/'non_int.h5ad',non_integer,GENES,{'Patient':qid})
    must_fail(lambda:score(root/'non_int.h5ad',root/'non_int.csv'),'integer-like')
    print('PASS 10: conflicting labels and non-count input rejected')

    # CLI create and score: no Python API code required by end users.
    cli_dir=root/'cli_model'
    cmd_create=[sys.executable,'-m','nsclc_regimen_mil.custom','create',
                '--input',str(root/'train.h5ad'),'--out-dir',str(cli_dir),
                '--layer','counts','--epochs','1','--cells-per-bag','8','--pool-cells','16','--device','cpu']
    env=dict(os.environ);env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
    subprocess.run(cmd_create,env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    cli_csv=root/'cli_output.csv'
    cmd_score=[sys.executable,'-m','nsclc_regimen_mil.custom','score',
               '--input',str(root/'novel.h5ad'),'--model-dir',str(cli_dir),
               '--patient-col','Patient','--layer','counts','--max-cells','10',
               '--repeats','3','--output',str(cli_csv),'--device','cpu']
    subprocess.run(cmd_score,env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    assert len(pd.read_csv(cli_csv))==4
    assert len(pd.read_csv(root/'cli_output_repeats.csv'))==12
    print('PASS 11: command line create and score produce valid output for unseen patients')
    again=score(root/'novel.h5ad',root/'again.csv')
    again=again.sort_values(['Patient','Regimen']).reset_index(drop=True)
    assert np.array_equal(a.Exact_RminusNR.to_numpy(),again.Exact_RminusNR.to_numpy())
    in_x=predict_custom(root/'novel.h5ad',model_dir,root/'in_x.csv',layer='X',
                        patient_col='Patient',max_cells=10,repeats=3,seed=72,device='cpu')
    in_x=in_x.sort_values(['Patient','Regimen']).reset_index(drop=True)
    assert np.array_equal(a.Exact_RminusNR.to_numpy(),in_x.Exact_RminusNR.to_numpy())
    print('PASS 12: deterministic repeat with same seed and raw counts in X vs layers/counts')
    print('\nALL 12 END-TO-END CHECKS PASSED (synthetic inputs; not biological model validation)')
    print(standard.to_string(index=False,float_format=lambda x:f'{x:.6f}'))

if __name__=='__main__':
    with TemporaryDirectory(prefix='regimen_mil_e2e_') as t:
        main(t)
