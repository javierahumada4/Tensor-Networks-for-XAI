"""Full raw-interaction decomposition on small trained ADBench models."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from data_artifacts import load_encoded_bundle
from mps import MPS
from mps_interactions import interaction_count, order_fidelity_curve, raw_interactions

def q(v,p):
    return float(np.quantile(np.asarray(v,dtype=float),p))

def fast_sparsity_curve(score, interactions, eps=1e-12):
    den=max(abs(float(score)),eps)
    pos=np.asarray(sorted((float(v) for v in interactions.values() if v>0),reverse=True),dtype=np.float64)
    neg=np.asarray(sorted(float(v) for v in interactions.values() if v<0),dtype=np.float64)
    pp=np.concatenate(([0.0],np.cumsum(pos,dtype=np.float64)))
    nn=np.concatenate(([0.0],np.cumsum(neg,dtype=np.float64)))
    rows=[]
    for k in range(max(len(pos),len(neg))+1):
        kp,kn=min(k,len(pos)),min(k,len(neg))
        recon=float(pp[kp]+nn[kn])
        residual=float(score)-recon
        rows.append({"k_per_sign":k,"num_interactions":kp+kn,"reconstruction":recon,
                     "signed_residual":residual,"c_k":abs(residual)/den})
    return rows

def summarize_fidelity(df):
    rows=[]
    for order,g in df.groupby("order",sort=True):
        vals=g.c_m.astype(float).to_numpy()
        contrib=(g.order_contribution.astype(float)/g.anomaly_score.astype(float)).to_numpy()
        rows.append({"order":int(order),"n":len(g),
          "median_c_m":float(np.median(vals)),"q25_c_m":q(vals,.25),"q75_c_m":q(vals,.75),
          "mean_c_m":float(np.mean(vals)),
          "median_order_contribution_over_score":float(np.median(contrib)),
          "q25_order_contribution_over_score":q(contrib,.25),
          "q75_order_contribution_over_score":q(contrib,.75),
          "fraction_le_0_10":float(np.mean(vals<=.10)),
          "fraction_le_0_05":float(np.mean(vals<=.05))})
    return pd.DataFrame(rows)

def summarize_sparsity(df):
    rows=[]
    for k,g in df.groupby("k_per_sign",sort=True):
        vals=g.c_k.astype(float).to_numpy()
        sizes=g.num_interactions.astype(int).to_numpy()
        rows.append({"k_per_sign":int(k),"n":len(g),
          "median_num_interactions":float(np.median(sizes)),
          "median_c_k":float(np.median(vals)),
          "q25_c_k":q(vals,.25),"q75_c_k":q(vals,.75)})
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("dataset_root",type=Path)
    p.add_argument("dataset")
    p.add_argument("model_dir",type=Path)
    p.add_argument("output_dir",type=Path)
    p.add_argument("--max-samples",type=int,default=50)
    p.add_argument("--seed",type=int,default=123)
    a=p.parse_args()

    data=load_encoded_bundle(a.dataset_root/a.dataset)
    model=MPS.load(str(a.model_dir/"model.pt"),map_location="cpu")
    model.eval()
    d=model.num_sites
    if d!=len(data.feature_names):
        raise ValueError("feature mismatch")

    all_scores=model.anomaly_score(data.test.x,batch_size=1024).detach().cpu().numpy()
    y=data.test.y.detach().cpu().numpy().astype(int)
    auroc=float(roc_auc_score(y,all_scores))
    auprc=float(average_precision_score(y,all_scores))
    precision,recall,_=precision_recall_curve(y,all_scores)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    max_f1=float(np.nanmax(f1))

    anomalies=np.flatnonzero(y==1)
    rng=np.random.default_rng(a.seed)
    n=min(a.max_samples,len(anomalies))
    selected=np.sort(anomalies if n==len(anomalies) else rng.choice(anomalies,size=n,replace=False))

    a.output_dir.mkdir(parents=True,exist_ok=True)
    frows=[]; srows=[]; sample_rows=[]
    full_count=interaction_count(d,d)

    for si,tp in enumerate(selected,1):
        x=data.test.x[int(tp)]
        score=float(model.anomaly_score(x.unsqueeze(0))[0].item())
        inter=raw_interactions(model,x,max_order=d)
        if len(inter)!=full_count:
            raise RuntimeError("incomplete decomposition")
        fid=order_fidelity_curve(score,inter,max_order=d)
        spa=fast_sparsity_curve(score,inter)
        common={"dataset":a.dataset,"sample_index":si,"test_position":int(tp),"anomaly_score":score}
        frows.extend({**common,**row} for row in fid)
        srows.extend({**common,**row} for row in spa)
        sample_rows.append({**common,"num_features":d,"num_interactions":full_count,"c_full":fid[-1]["c_m"]})
        print(f"[{a.dataset} {si}/{n}] d={d} subsets={full_count} c_d={fid[-1]['c_m']:.3e}",flush=True)

    fdf=pd.DataFrame(frows)
    sdf=pd.DataFrame(srows)
    fsum=summarize_fidelity(fdf)
    ssum=summarize_sparsity(sdf)
    pd.DataFrame(sample_rows).to_csv(a.output_dir/"samples.csv",index=False)
    fdf.to_csv(a.output_dir/"fidelity_per_sample.csv",index=False)
    sdf.to_csv(a.output_dir/"sparsity_per_sample.csv",index=False)
    fsum.to_csv(a.output_dir/"fidelity_summary.csv",index=False)
    ssum.to_csv(a.output_dir/"sparsity_summary.csv",index=False)

    meta={"dataset":a.dataset,"seed":a.seed,"sample_count":int(n),
      "num_features":int(d),"max_order":int(d),"full_decomposition":True,
      "interaction_subsets_per_sample":int(full_count),"feature_names":data.feature_names,
      "detection":{"auroc":auroc,"auprc":auprc,"max_f1":max_f1},
      "closure":{"median_c_d":float(fsum.iloc[-1].median_c_m),
                 "max_c_d":float(fdf[fdf.order==d].c_m.astype(float).max())}}
    (a.output_dir/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps(meta,indent=2),flush=True)

if __name__=="__main__":
    main()
