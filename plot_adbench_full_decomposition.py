"""Aggregate and plot complete ADBench interaction decompositions."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER=["annthyroid","mammography","shuttle","cover","vowels"]

def find_base(root,name):
    for pat in [f"**/adbench-full-{name}/metadata.json", f"**/{name}/metadata.json"]:
        xs=list(root.glob(pat))
        if xs:
            return xs[0].parent
    raise RuntimeError(f"missing {name}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("output_dir",type=Path)
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)

    ds={}
    meta_rows=[]
    for name in ORDER:
        base=find_base(a.root,name)
        meta=json.loads((base/"metadata.json").read_text())
        fid=pd.read_csv(base/"fidelity_summary.csv")
        spa=pd.read_csv(base/"sparsity_summary.csv")
        ds[name]=(meta,fid,spa)
        meta_rows.append({
            "dataset":name,
            "features":meta["num_features"],
            "full_interactions":meta["interaction_subsets_per_sample"],
            "samples":meta["sample_count"],
            "auroc":meta["detection"]["auroc"],
            "auprc":meta["detection"]["auprc"],
            "max_f1":meta["detection"]["max_f1"],
            "median_c_full":meta["closure"]["median_c_d"],
            "max_c_full":meta["closure"]["max_c_d"],
        })
    pd.DataFrame(meta_rows).to_csv(a.output_dir/"dataset_summary.csv",index=False)

    fig,axes=plt.subplots(2,3,figsize=(14,8.5))
    axes=axes.ravel()
    for ax,name in zip(axes,ORDER):
        meta,fid,_=ds[name]
        y=np.maximum(fid.median_c_m.to_numpy(float),1e-16)
        lo=np.maximum(fid.q25_c_m.to_numpy(float),1e-16)
        hi=np.maximum(fid.q75_c_m.to_numpy(float),1e-16)
        ax.plot(fid.order,y,marker="o")
        ax.fill_between(fid.order,lo,hi,alpha=.18)
        ax.axhline(.10,linestyle=":",linewidth=1)
        ax.axhline(.05,linestyle=":",linewidth=1)
        ax.set_yscale("log")
        ax.set_xticks(fid.order)
        ax.set_title(f"{name}: d={meta['num_features']}, 2^d-1={meta['interaction_subsets_per_sample']}")
        ax.set_xlabel("Maximum interaction order m")
        ax.set_ylabel("Median relative residual c_m")
    axes[-1].axis("off")
    fig.suptitle("Full interaction decomposition: fidelity through maximum order")
    fig.tight_layout()
    fig.savefig(a.output_dir/"full_fidelity_by_order.png",dpi=220)
    fig.savefig(a.output_dir/"full_fidelity_by_order.pdf")
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.2,5.2))
    for name in ORDER:
        meta,fid,_=ds[name]
        x=fid.order.to_numpy(float)/meta["num_features"]
        y=np.maximum(fid.median_c_m.to_numpy(float),1e-16)
        ax.plot(x,y,marker="o",label=f"{name} (d={meta['num_features']})")
    ax.set_yscale("log")
    ax.axhline(.10,linestyle=":",linewidth=1)
    ax.axhline(.05,linestyle=":",linewidth=1)
    ax.set_xlabel("Normalized interaction order m/d")
    ax.set_ylabel("Median relative residual c_m")
    ax.set_title("Complete decomposition: all curves close at m=d")
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.output_dir/"normalized_order_comparison.png",dpi=220)
    fig.savefig(a.output_dir/"normalized_order_comparison.pdf")
    plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(14,8.5))
    axes=axes.ravel()
    for ax,name in zip(axes,ORDER):
        _,fid,_=ds[name]
        ax.axhline(0,linewidth=1)
        ax.plot(fid.order,fid.median_order_contribution_over_score,marker="o")
        ax.fill_between(fid.order,fid.q25_order_contribution_over_score,
                        fid.q75_order_contribution_over_score,alpha=.18)
        ax.set_xticks(fid.order)
        ax.set_title(name)
        ax.set_xlabel("Interaction order")
        ax.set_ylabel("Median order contribution / A(x)")
    axes[-1].axis("off")
    fig.suptitle("Signed contribution of each interaction order")
    fig.tight_layout()
    fig.savefig(a.output_dir/"order_contributions.png",dpi=220)
    fig.savefig(a.output_dir/"order_contributions.pdf")
    plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(14,8.5))
    axes=axes.ravel()
    for ax,name in zip(axes,ORDER):
        _,_,spa=ds[name]
        g=spa[spa.median_num_interactions>0]
        x=g.median_num_interactions.to_numpy(float)
        y=np.maximum(g.median_c_k.to_numpy(float),1e-16)
        ax.plot(x,y)
        ax.axhline(.10,linestyle=":",linewidth=1)
        ax.axhline(.05,linestyle=":",linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("Median retained interactions")
        ax.set_ylabel("Median relative residual c_k")
    axes[-1].axis("off")
    fig.suptitle("Full-decomposition sparsity curves")
    fig.tight_layout()
    fig.savefig(a.output_dir/"full_sparsity_curves.png",dpi=220)
    fig.savefig(a.output_dir/"full_sparsity_curves.pdf")
    plt.close(fig)

    report={"datasets":meta_rows,
            "note":"Every non-empty subset is computed through order d, so c_d closes to numerical zero independently of detector quality."}
    (a.output_dir/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
