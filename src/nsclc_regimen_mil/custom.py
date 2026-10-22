"""Experimental pan-cancer training and Exact R-NR inference for real scRNA-seq.

Create a model from *labelled patients* rather than reusing the NSCLC checkpoint.
The resulting scores are exploratory cosine contrasts, not treatment benefits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .model import TreatmentConditionedMultiHeadAttentionMIL


def _str_array(values):
    return np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v)
                       for v in values], dtype=str)


def _read_field(group, field):
    if field not in group:
        raise KeyError(f"Missing H5AD field: {group.name}/{field}")
    node = group[field]
    if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
        codes = np.asarray(node["codes"], dtype=int)
        categories = _str_array(node["categories"][:])
        return np.asarray([categories[i] if i >= 0 else "" for i in codes])
    if isinstance(node, h5py.Dataset):
        return _str_array(node[:])
    raise ValueError(f"Unsupported H5AD field representation: {node.name}")


class H5ADCells:
    """Minimal read-only H5AD CSR/dense reader, avoiding backed fancy-index bugs."""

    def __init__(self, path, layer):
        self.file = h5py.File(path, "r")
        self.node = self.file["X" if layer == "X" else f"layers/{layer}"]
        var_group = self.file["var"]
        var_index_key = var_group.attrs.get("_index", "_index")
        if isinstance(var_index_key, bytes):
            var_index_key = var_index_key.decode("utf-8")
        self.genes = pd.Index(_read_field(var_group, str(var_index_key)))
        if not self.genes.is_unique:
            raise ValueError("Duplicate gene identifiers; fix var_names first")
        self.n_cells = int(self.node.attrs["shape"][0]) if isinstance(self.node, h5py.Group) else self.node.shape[0]
        self.n_genes = len(self.genes)
        if isinstance(self.node, h5py.Group):
            self.shape = tuple(int(x) for x in self.node.attrs["shape"])
            encoding = self.node.attrs.get("encoding-type", "")
            if isinstance(encoding, bytes):
                encoding = encoding.decode()
            if encoding != "csr_matrix":
                raise ValueError(f"This MVP requires CSR H5AD matrices; found {encoding}. Convert to CSR first")
            self.indptr = self.node["indptr"]
            self.indices = self.node["indices"]
            self.values = self.node["data"]
        else:
            self.shape = self.node.shape
        if self.shape[1] != self.n_genes:
            raise ValueError("Gene count mismatch between H5AD X and var index")

    def obs(self, field):
        v = _read_field(self.file["obs"], field)
        if len(v) != self.n_cells:
            raise ValueError(f"Incorrect obs length for {field}")
        return v

    def rows(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices):
            raise ValueError("Empty cell selection")
        if np.any(indices < 0) or np.any(indices >= self.n_cells):
            raise ValueError("Cell index outside H5AD")
        if isinstance(self.node, h5py.Dataset):
            unique, inv = np.unique(indices, return_inverse=True)
            return sp.csr_matrix(self.node[unique, :][inv].astype(np.float32))
        starts = self.indptr[indices]
        ends = self.indptr[indices + 1]
        lengths = np.asarray(ends - starts, dtype=np.int64)
        ptr = np.concatenate(([0], np.cumsum(lengths)))
        n_nz = int(ptr[-1])
        col = np.empty(n_nz, dtype=np.int32)
        val = np.empty(n_nz, dtype=np.float32)
        for i, (start, end) in enumerate(zip(starts, ends)):
            if end > start:
                col[ptr[i]:ptr[i+1]] = self.indices[int(start):int(end)]
                val[ptr[i]:ptr[i+1]] = self.values[int(start):int(end)]
        result = sp.csr_matrix((val, col, ptr), shape=(len(indices), self.n_genes))
        if result.nnz and (not np.isfinite(result.data).all() or (result.data < 0).any()):
            raise ValueError("Input contains negative or nonfinite counts")
        return result

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _normalize(counts, target_sum):
    x = counts.copy().tocsr().astype(np.float32)
    if x.nnz and not np.allclose(x.data, np.round(x.data), atol=1e-5):
        raise ValueError("Training/prediction expects raw, integer-like UMI counts")
    size = np.asarray(x.sum(axis=1)).reshape(-1)
    if np.any(size <= 0):
        raise ValueError("Zero-library cell found; QC input first")
    x = x.multiply((float(target_sum) / size)[:, None]).tocsr()
    x.data = np.log1p(x.data).astype(np.float32)
    return x


def _to_torch(x, device):
    coo = x.tocoo()
    indices = torch.as_tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
    vals = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, vals, coo.shape, device=device).coalesce()


def _seed(seed, identity):
    return int(hashlib.sha256(f"{seed}|{identity}".encode()).hexdigest()[:8], 16)


def _patient_catalog(reader, patient_col, treatment_col, response_col):
    patients = reader.obs(patient_col)
    treatments = reader.obs(treatment_col)
    responses = reader.obs(response_col)
    catalog = {}
    for p in sorted(set(patients)):
        if not p or p.lower() in ("nan", "none"):
            raise ValueError("Missing patient ID")
        idx = np.flatnonzero(patients == p)
        treatment = set(treatments[idx])
        response = set(responses[idx])
        if len(treatment) != 1 or len(response) != 1:
            raise ValueError(f"Patient {p} has conflicting treatment/response labels")
        t = treatment.pop()
        r = response.pop().strip().upper()
        label = {"R": 1, "NR": 0, "1": 1, "0": 0, "RESPONDER": 1, "NON-RESPONDER": 0}.get(r)
        if not t or label is None:
            raise ValueError(f"Patient {p}: unknown treatment or response {r!r}")
        catalog[p] = {"rows": idx, "treatment": t, "label": label}
    return catalog


def _unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(v)
    if norm < 1e-12 or not np.isfinite(norm):
        raise ValueError("Invalid zero/nonfinite patient embedding")
    return v / norm


@torch.no_grad()
def _embedding(model, matrix, treatment_index, device):
    model.eval()
    h = model.encode_cells(_to_torch(matrix, device))
    logit, _, heads = model.patient_logit_from_h(h, treatment_index, return_attention=True)
    return float(torch.sigmoid(logit).item()), _unit(heads.detach().cpu().numpy())


def create_model(
    train_h5ad, output_dir, *, patient_col="Patient", treatment_col="Treatment",
    response_col="response_binary", layer="counts", target_sum=10000,
    epochs=35, cells_per_bag=256, pool_cells=1024, seed=42,
    learning_rate=2e-4, device=None,
):
    """Train a *new* regimen-MIL model and same-model R/NR reference means.

    This is a fitting API, not a patient-held-out validation estimator.
    """
    if epochs < 1 or cells_per_bag < 1 or pool_cells < cells_per_bag or not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("Invalid epochs, cell limits, or target_sum")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with H5ADCells(train_h5ad, layer) as reader:
        patients = _patient_catalog(reader, patient_col, treatment_col, response_col)
        regimens = sorted(str(r) for r in {info["treatment"] for info in patients.values()})
        if len(patients) < 8:
            raise ValueError("At least 8 labelled patients required for this experimental training API")
        for r in regimens:
            labels = [v["label"] for v in patients.values() if v["treatment"] == r]
            if labels.count(0) < 2 or labels.count(1) < 2:
                raise ValueError(f"Regimen {r!r}: require >=2 independent R and >=2 independent NR patients")
        indices = {r: i for i, r in enumerate(regimens)}
        gene_order = reader.genes.tolist()
        print(f"Training: {len(patients)} patients, {len(regimens)} regimens, {len(gene_order)} genes", flush=True)
        print("Note: training metrics are not external validation", flush=True)

        # Cache a reproducible candidate cell pool per patient; sample sub-bags each epoch.
        cached = {}
        for p, info in patients.items():
            ids = info["rows"]
            if len(ids) > pool_cells:
                ids = np.sort(np.random.default_rng(_seed(seed, p)).choice(ids, pool_cells, replace=False))
            cached[p] = _normalize(reader.rows(ids), target_sum)
        cfg = dict(n_genes=len(gene_order), n_treatments=len(regimens),
                   n_attention_heads=4, cell_embed_dim=128, treatment_embed_dim=16,
                   attention_dim=64, classifier_hidden=64,
                   cell_dropout=0.2, patient_dropout=0.3)
        model = TreatmentConditionedMultiHeadAttentionMIL(**cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        names = np.array(list(patients))
        losses = []
        for epoch in range(epochs):
            model.train()
            running = 0.0
            for p in rng.permutation(names):
                x = cached[p]
                if x.shape[0] > cells_per_bag:
                    chosen = np.sort(rng.choice(x.shape[0], cells_per_bag, replace=False))
                    x = x[chosen, :]
                info = patients[p]
                y = torch.tensor(float(info["label"]), device=device)
                optimizer.zero_grad(set_to_none=True)
                logit = model.forward_bag(_to_torch(x, device), indices[info["treatment"]])
                loss = F.binary_cross_entropy_with_logits(logit.view(()), y)
                loss.backward()
                optimizer.step()
                running += float(loss.detach().item())
            losses.append(running / len(patients))
            print(f"Epoch {epoch + 1:03d}/{epochs}: training BCE={losses[-1]:.5f}", flush=True)

        ck_file = out / "custom_mil.pt"
        torch.save({"state_dict": model.cpu().state_dict(), "config": cfg,
                    "regimens": regimens, "target_sum": float(target_sum),
                    "train_patient_count": len(patients),
                    "gene_order_sha256": hashlib.sha256("\n".join(gene_order).encode("utf-8")).hexdigest()}, ck_file)
        model.to(device)
        digest = hashlib.sha256(ck_file.read_bytes()).hexdigest()

        # Make reference means from all observed same-regimen labelled patients.
        # Unit vectors are averaged WITHOUT renormalising means, preserving Exact R-NR.
        reference = {(r, y): [] for r in regimens for y in (0, 1)}
        audit = []
        for p, info in patients.items():
            t = info["treatment"]
            _, vector = _embedding(model, cached[p], indices[t], device)
            reference[(t, info["label"])].append(vector)
            audit.append({"Patient": p, "Treatment": t, "response_binary": info["label"],
                          "N_cells_cached": cached[p].shape[0]})
        means_R = np.stack([np.mean(reference[(r, 1)], axis=0) for r in regimens])
        means_NR = np.stack([np.mean(reference[(r, 0)], axis=0) for r in regimens])
        np.savez_compressed(out / "reference_centroids.npz", regimens=np.array(regimens),
                            R=means_R, NR=means_NR,
                            N_R=np.array([len(reference[(r, 1)]) for r in regimens]),
                            N_NR=np.array([len(reference[(r, 0)]) for r in regimens]),
                            checkpoint_sha256=np.array(digest))
        pd.Series(gene_order, name="gene").to_csv(out / "gene_order.csv", index=False)
        pd.DataFrame({"treatment_index": range(len(regimens)), "Treatment": regimens}).to_csv(
            out / "treatment_vocabulary.csv", index=False)
        pd.DataFrame(audit).to_csv(out / "reference_coverage.csv", index=False)
        with open(out / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"input": str(train_h5ad), "patient_col": patient_col,
                       "treatment_col": treatment_col, "response_col": response_col,
                       "layer": layer, "target_sum": target_sum, "epochs": epochs,
                       "pool_cells": pool_cells, "cells_per_bag": cells_per_bag,
                       "seed": seed, "training_loss": losses,
                       "warning": "No held-out or cross-validated validation was run"}, f, indent=2)
    print(f"Model, gene order and reference bank saved to: {out}", flush=True)
    return out


def predict_custom(query_h5ad, model_dir, output_csv, *, patient_col="Patient", patient_id=None,
                   layer="counts", max_cells=1024, repeats=3, seed=42,
                   min_gene_coverage=0.8, device=None):
    """Score unseen patient bags against the *matching* custom reference bank."""
    if repeats < 1 or max_cells < 1 or not 0 < min_gene_coverage <= 1:
        raise ValueError("Invalid repetitions, cell count, or coverage")
    directory = Path(model_dir)
    ck_file = directory / "custom_mil.pt"
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(ck_file, weights_only=True, map_location="cpu")
    genes = pd.Index(pd.read_csv(directory / "gene_order.csv")["gene"].astype(str))
    regimens = ck["regimens"]
    gene_hash = hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest()
    if ck.get("gene_order_sha256") != gene_hash:
        raise ValueError("Gene-order file does not match the trained checkpoint")
    if len(genes) != ck["config"]["n_genes"] or not genes.is_unique:
        raise ValueError("Invalid or duplicated gene identifiers in gene_order.csv")
    if len(regimens) != ck["config"]["n_treatments"]:
        raise ValueError("Treatment vocabulary and checkpoint mismatch")
    model = TreatmentConditionedMultiHeadAttentionMIL(**ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    with np.load(directory / "reference_centroids.npz", allow_pickle=False) as bank:
        if bank["regimens"].astype(str).tolist() != regimens:
            raise ValueError("Regimen vocabulary mismatch")
        if str(bank["checkpoint_sha256"].item()) != hashlib.sha256(ck_file.read_bytes()).hexdigest():
            raise ValueError("Reference bank checkpoint mismatch")
        R = np.asarray(bank["R"])
        NR = np.asarray(bank["NR"])
        n_r = np.asarray(bank["N_R"])
        n_nr = np.asarray(bank["N_NR"])
    # Do not accidentally score historical training patients as independent queries.
    audit_file = directory / "reference_coverage.csv"
    if not audit_file.is_file():
        raise ValueError("Missing reference_coverage.csv; cannot check train/query overlap")
    training_ids = set(pd.read_csv(audit_file, dtype={"Patient": str},
                                   keep_default_na=False)["Patient"].tolist())
    results = []
    with H5ADCells(query_h5ad, layer) as reader:
        indexer = reader.genes.get_indexer(genes)
        coverage = float(np.mean(indexer >= 0))
        print(f"Matched genes: {(indexer >= 0).sum()}/{len(genes)} ({coverage:.1%})")
        if coverage < min_gene_coverage:
            raise ValueError("Insufficient gene coverage; check identifiers and missing genes")
        if patient_id is not None:
            if patient_col is not None and patient_col in reader.file["obs"]:
                raise ValueError("Use either --patient-col or --patient-id, not both")
            ids = np.full(reader.n_cells, str(patient_id))
        elif patient_col is None:
            ids = np.full(reader.n_cells, Path(query_h5ad).stem)
        else:
            ids = reader.obs(patient_col)
        overlap = training_ids.intersection(set(ids))
        if overlap:
            raise ValueError("Query contains training/reference patient IDs: "
                             + ", ".join(sorted(overlap)[:10])
                             + ". Use only genuinely new patients for inference")
        for patient in sorted(set(ids)):
            if not patient or patient.lower() in ("none", "nan"):
                raise ValueError("Missing patient ID")
            rows = np.flatnonzero(ids == patient)
            for repeat in range(repeats):
                if len(rows) > max_cells:
                    rows_selected = np.sort(np.random.default_rng(_seed(seed, f"{patient}-{repeat}"))
                                            .choice(rows, max_cells, replace=False))
                else:
                    rows_selected = rows
                # Normalise over the FULL query gene space before aligning to model genes.
                x = _normalize(reader.rows(rows_selected), ck["target_sum"])
                existing = np.flatnonzero(indexer >= 0)
                selected = x[:, indexer[existing]].tocoo()
                x_aligned = sp.coo_matrix(
                    (selected.data, (selected.row, existing[selected.col])),
                    shape=(x.shape[0], len(genes)), dtype=np.float32).tocsr()
                for t, regimen in enumerate(regimens):
                    raw, q = _embedding(model, x_aligned, t, device)
                    a, b = float(q @ R[t]), float(q @ NR[t])
                    results.append({"Patient": patient, "Regimen": regimen,
                                    "repeat": repeat, "N_cells": len(rows_selected),
                                    "raw_R_score": raw,
                                    "Mean_R_similarity": a, "Mean_NR_similarity": b,
                                    "Exact_RminusNR": a - b,
                                    "Reference_R": int(n_r[t]), "Reference_NR": int(n_nr[t])})
    if not results:
        raise ValueError("No query patients or scores were generated")
    raw = pd.DataFrame(results)
    summary = raw.groupby(["Patient", "Regimen"], as_index=False).agg(
        Exact_RminusNR=("Exact_RminusNR", "mean"),
        Score_SD=("Exact_RminusNR", "std"),
        Mean_R_similarity=("Mean_R_similarity", "mean"),
        Mean_NR_similarity=("Mean_NR_similarity", "mean"),
        raw_R_score=("raw_R_score", "mean"),
        Reference_R=("Reference_R", "first"), Reference_NR=("Reference_NR", "first"))
    summary["Score_SD"] = summary["Score_SD"].fillna(0.)
    summary["Exact_rank"] = summary.groupby("Patient")["Exact_RminusNR"].rank(
        method="first", ascending=False).astype(int)
    summary.sort_values(["Patient", "Exact_rank"], inplace=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    raw.to_csv(output_csv.with_name(output_csv.stem + "_repeats.csv"), index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train a new cancer-specific regimen MIL model or score new patients")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("create", help="Train cancer-specific MIL and reference bank")
    train.add_argument("--input", required=True)
    train.add_argument("--out-dir", required=True)
    train.add_argument("--patient-col", default="Patient")
    train.add_argument("--treatment-col", default="Treatment")
    train.add_argument("--response-col", default="response_binary")
    train.add_argument("--layer", default="counts")
    train.add_argument("--target-sum", type=float, default=10000)
    train.add_argument("--epochs", type=int, default=35)
    train.add_argument("--cells-per-bag", type=int, default=256)
    train.add_argument("--pool-cells", type=int, default=1024)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default=None)
    predict = commands.add_parser("score", help="Score new real scRNA-seq with a custom model")
    predict.add_argument("--input", required=True)
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--patient-col", default="Patient")
    predict.add_argument("--patient-id", default=None,
                         help="Single patient ID when H5AD has no patient column")
    predict.add_argument("--layer", default="counts")
    predict.add_argument("--max-cells", type=int, default=1024)
    predict.add_argument("--repeats", type=int, default=3)
    predict.add_argument("--min-gene-coverage", type=float, default=0.8)
    predict.add_argument("--seed", type=int, default=42)
    predict.add_argument("--device", default=None)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "create":
        create_model(train_h5ad=args.pop("input"), output_dir=args.pop("out_dir"), **args)
    else:
        predict_custom(query_h5ad=args.pop("input"), model_dir=args.pop("model_dir"),
                       output_csv=args.pop("output"), **args)


if __name__ == "__main__":
    main()
