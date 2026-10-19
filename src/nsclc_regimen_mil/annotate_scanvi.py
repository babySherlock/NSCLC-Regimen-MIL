"""Optional query-to-reference scANVI annotation. Does not train or modify MIL."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scvi
import torch


def main():
    parser = argparse.ArgumentParser(description="Map raw-count scRNA query into scANVI labels")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/scanvi_reference"))
    parser.add_argument("--patient-col", default=None)
    parser.add_argument("--layer", default="X")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("outputs/cell_annotations.csv"))
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input)
    if args.patient_col is None:
        adata.obs["batch_id"] = "Query::" + args.input.stem
    else:
        if args.patient_col not in adata.obs.columns:
            raise KeyError(f"obs[{args.patient_col}] not found")
        adata.obs["batch_id"] = "Query::" + adata.obs[args.patient_col].astype(str)
    if args.layer == "X":
        counts = adata.X
    else:
        if args.layer not in adata.layers:
            raise KeyError(f"Query layer {args.layer} missing")
        counts = adata.layers[args.layer]
    values = counts.data if sp.issparse(counts) else np.asarray(counts).reshape(-1)
    if values.size and (
        (values < 0).any() or not np.isfinite(values).all()
        or not np.allclose(values, np.rint(values), atol=1e-5)
    ):
        raise ValueError("scANVI must receive raw, nonnegative integer-like UMI counts")
    adata.layers["counts"] = counts.copy()
    adata.obs["scanvi_labels"] = "Unknown"

    scvi.model.SCANVI.prepare_query_anndata(adata, str(args.model_dir))
    mapped = scvi.model.SCANVI.load_query_data(adata, str(args.model_dir))
    mapped.train(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        plan_kwargs={"weight_decay": 0.0},
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    prob = mapped.predict(adata, soft=True, batch_size=args.batch_size)
    if not isinstance(prob, pd.DataFrame):
        raise TypeError("scANVI soft prediction should be DataFrame; check label ordering")
    prob = prob.reindex(adata.obs_names)
    if prob.isna().any().any() or prob.shape[1] < 2:
        raise ValueError("Invalid scANVI prediction matrix")
    a = prob.to_numpy(dtype=np.float32)
    top1 = a.max(axis=1)
    top2 = np.partition(a, -2, axis=1)[:, -2]
    labels = prob.idxmax(axis=1).astype(str).to_numpy()
    accepted = (top1 >= 0.70) & ((top1 - top2) >= 0.15)
    result = pd.DataFrame({
        "barcode": adata.obs_names.astype(str),
        "predicted_celltype": labels,
        "scanvi_pmax": top1,
        "scanvi_margin": top1 - top2,
        "scanvi_confident_label": np.where(accepted, labels, "Unknown"),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result["scanvi_confident_label"].value_counts())
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
