"""Retrieve same-regimen historical patients with cosine similarity and SCENIC evidence.

Read-only, inference-only: never retrains MIL, scANVI, or pySCENIC.
Data files represent derived patient-level research records and may need private distribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ATLAS_FILES = (
    "reference_patient_embeddings.npz",
    "reference_patient_regulons.csv.gz",
    "reference_regulon_targets.csv.gz",
    "atlas_manifest_patientwise.json",
)
AUC_KEYS = ["sampleID", "Treatment", "Response", "Subtype", "Regulon", "TF", "AUCell"]
EDGE_KEYS = ["Subtype", "Regulon", "TF", "Target"]


def _require_columns(df, columns, context):
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"{context}: missing columns {missing}")


def _normalize_rows(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Reference embeddings must be a nonempty 2D array")
    if not np.isfinite(x).all():
        raise ValueError("Reference embeddings contain NaN or inf")
    norms = np.linalg.norm(x, axis=1)
    if (norms <= 1e-10).any():
        raise ValueError("Reference contains zero-norm embeddings")
    return x / norms[:, None]


def load_reference(atlas_dir, checkpoint_sha256, expected_patients=156):
    """Validate the patientwise notebook's complete output; preserve row correspondence."""
    atlas_dir = Path(atlas_dir)
    absent = [x for x in ATLAS_FILES if not (atlas_dir / x).is_file()]
    if absent:
        raise FileNotFoundError(f"Atlas incomplete at {atlas_dir}: missing {absent}")
    manifest = json.loads((atlas_dir / "atlas_manifest_patientwise.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete"
            or manifest.get("historical_patients_expected") != expected_patients
            or manifest.get("historical_patients_auc_done") != expected_patients
            or manifest.get("historical_patients_embedding_done") != expected_patients):
        raise ValueError("Atlas manifest is not a verified complete 156-patient atlas")
    if manifest.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Atlas manifest MIL checkpoint SHA256 mismatch")
    with np.load(atlas_dir / "reference_patient_embeddings.npz", allow_pickle=False) as z:
        expected = {"embeddings", "patient_ids", "actual_regimens", "responses", "checkpoint_sha256"}
        if not expected.issubset(z.files):
            raise ValueError(f"Embedding archive missing {sorted(expected - set(z.files))}")
        if str(z["checkpoint_sha256"].item()) != checkpoint_sha256:
            raise ValueError("Embedding archive checkpoint SHA256 mismatch")
        embs = _normalize_rows(z["embeddings"])
        meta = pd.DataFrame({
            "Patient": z["patient_ids"].astype(str),
            "Regimen": z["actual_regimens"].astype(str),
            "Response": z["responses"].astype(str),
        })
    if len(meta) != expected_patients or len(embs) != expected_patients:
        raise ValueError("Atlas patient count mismatch")
    if not meta.Patient.is_unique or meta.Patient.eq("").any():
        raise ValueError("Patient identifiers must be unique and nonempty")
    if not meta.Response.isin(["R", "NR"]).all():
        raise ValueError("Reference outcomes must be R or NR")
    if meta.Regimen.eq("").any():
        raise ValueError("Reference treatment cannot be empty")
    return embs, meta, manifest


def cosine_neighbors(query_vector, embeddings, metadata, regimen, query_patient=None):
    """Cosine within *actually received* historical treatment only, no label mixing."""
    vec = np.asarray(query_vector, dtype=np.float64).reshape(-1)
    if not np.isfinite(vec).all() or np.linalg.norm(vec) < 1e-10:
        raise ValueError("Invalid query embedding")
    vec /= np.linalg.norm(vec)
    if embeddings.shape[1] != len(vec):
        raise ValueError("Query/reference embedding dimensions differ")
    mask = metadata.Regimen.eq(regimen).to_numpy()
    if query_patient is not None:
        mask &= metadata.Patient.ne(str(query_patient)).to_numpy()  # avoid self-match
    idx = np.flatnonzero(mask)
    ranked = metadata.iloc[idx].copy()
    ranked["cosine_similarity"] = np.clip(embeddings[idx] @ vec, -1., 1.)
    ranked = ranked.sort_values("cosine_similarity", ascending=False, kind="stable")
    ranked.insert(0, "neighbor_rank", np.arange(1, len(ranked) + 1))
    return ranked.reset_index(drop=True)


def load_regulons(atlas_dir, metadata):
    """Moderate-sized derived AUCell matrix, not raw cells; validates provenance."""
    atlas_dir = Path(atlas_dir)
    auc = pd.read_csv(atlas_dir / "reference_patient_regulons.csv.gz")
    _require_columns(auc, AUC_KEYS, "AUCell")
    auc["sampleID"] = auc.sampleID.astype(str)
    if not set(auc.sampleID).issubset(set(metadata.Patient)):
        raise ValueError("AUCell includes a patient outside the 156-patient atlas")
    auc["AUCell"] = pd.to_numeric(auc.AUCell, errors="raise")
    if not np.isfinite(auc.AUCell.to_numpy()).all():
        raise ValueError("Nonfinite AUCell values")
    key = ["sampleID", "Subtype", "Regulon"]
    if auc.duplicated(key).any():
        raise ValueError("Duplicate patient/subtype/regulon entries")
    chk = auc.merge(metadata, left_on="sampleID", right_on="Patient", how="left", validate="many_to_one")
    if (chk.Treatment.astype(str) != chk.Regimen.astype(str)).any() or (chk.Response_x.astype(str) != chk.Response_y.astype(str)).any():
        raise ValueError("AUCell regimen or R/NR disagrees with embedding archive")
    # Percentile is within each subtype/regulon across the available historical patients,
    # not an absolute measure of statistical significance.
    auc["AUCell_percentile"] = auc.groupby(["Subtype", "Regulon"])["AUCell"].rank(pct=True)
    edges = pd.read_csv(atlas_dir / "reference_regulon_targets.csv.gz")
    _require_columns(edges, EDGE_KEYS, "TF-target")
    edges = edges[EDGE_KEYS].dropna().drop_duplicates()
    return auc, edges


def summarize_tf_candidates(
    ranked, auc, edges, regimen, query_patient,
    neighbors_per_class=5, top_tfs=15, min_group_n=2,
):
    """Historical descriptive TF support, with matched-regimen R-NR contrast if estimable."""
    r = ranked.loc[ranked.Response.eq("R")].head(neighbors_per_class)
    nr = ranked.loc[ranked.Response.eq("NR")].head(neighbors_per_class)
    base = {
        "query_patient": str(query_patient), "Regimen": regimen,
        "n_historical_total": int(len(ranked)),
        "n_historical_R": int(ranked.Response.eq("R").sum()),
        "n_historical_NR": int(ranked.Response.eq("NR").sum()),
        "n_nearest_R": int(len(r)), "n_nearest_NR": int(len(nr)),
    }
    if r.empty:
        return pd.DataFrame(), pd.DataFrame(), {**base, "status": "NO_HISTORICAL_R_FOR_REGIMEN"}
    same = auc.loc[auc.Treatment.eq(regimen)].copy()
    if same.empty:
        return pd.DataFrame(), pd.DataFrame(), {**base, "status": "NO_SCENIC_EVIDENCE_FOR_REGIMEN"}
    # Descriptive differences among *historical patients who actually received the regimen*.
    keys = ["Subtype", "Regulon", "TF"]
    contrast = same.groupby(keys + ["Response"], observed=True).agg(
        mean_AUCell=("AUCell", "mean"), n_patients=("sampleID", "nunique"),
    ).reset_index()
    pivot_mean = contrast.pivot(index=keys, columns="Response", values="mean_AUCell")
    pivot_n = contrast.pivot(index=keys, columns="Response", values="n_patients")
    for label in ("R", "NR"):
        if label not in pivot_mean.columns: pivot_mean[label] = np.nan
        if label not in pivot_n.columns: pivot_n[label] = 0
    diff = pd.DataFrame({
        "Subtype": pivot_mean.index.get_level_values(0),
        "Regulon": pivot_mean.index.get_level_values(1),
        "TF": pivot_mean.index.get_level_values(2),
        "regimen_n_R_measured": pivot_n.R.fillna(0).astype(int).to_numpy(),
        "regimen_n_NR_measured": pivot_n.NR.fillna(0).astype(int).to_numpy(),
        "historical_R_minus_NR_AUCell": (pivot_mean.R - pivot_mean.NR).to_numpy(),
    })
    valid = (diff.regimen_n_R_measured >= min_group_n) & (diff.regimen_n_NR_measured >= min_group_n)
    diff.loc[~valid, "historical_R_minus_NR_AUCell"] = np.nan
    # Retrieve descriptive activity among nearest *R* neighbors, without falsely
    # requiring the top-ranked neighbours overall to be R.
    r_auc = same.loc[same.sampleID.isin(r.Patient)]
    if r_auc.empty:
        return pd.DataFrame(), pd.DataFrame(), {**base, "status": "NEAREST_R_WITHOUT_SCENIC_MEASUREMENTS"}
    r_summary = r_auc.groupby(keys, observed=True).agg(
        n_R_supporters=("sampleID", "nunique"),
        mean_R_neighbor_AUCell=("AUCell", "mean"),
        mean_R_neighbor_percentile=("AUCell_percentile", "mean"),
    ).reset_index()
    r_summary["R_support_fraction"] = r_summary.n_R_supporters / len(r)
    # Only include an NR-neighbour comparison when a measured NR exists.
    nr_auc = same.loc[same.sampleID.isin(nr.Patient)]
    if len(nr_auc):
        nr_sum = nr_auc.groupby(keys, observed=True).agg(
            n_NR_supporters=("sampleID", "nunique"),
            mean_NR_neighbor_AUCell=("AUCell", "mean"),
        ).reset_index()
        r_summary = r_summary.merge(nr_sum, on=keys, how="left", validate="one_to_one")
    else:
        r_summary["n_NR_supporters"] = 0
        r_summary["mean_NR_neighbor_AUCell"] = np.nan
    candidates = r_summary.merge(diff, on=keys, how="left", validate="one_to_one")
    candidates["candidate_basis"] = np.where(
        candidates.historical_R_minus_NR_AUCell.gt(0).fillna(False),
        "R_enriched_descriptive", "nearest_R_regulon_activity_only",
    )
    # Not a p value / causal score: supports high activity and coverage of R neighbours.
    candidates["retrieval_score"] = candidates.mean_R_neighbor_percentile * candidates.R_support_fraction
    candidates = candidates.sort_values(
        ["retrieval_score", "n_R_supporters", "mean_R_neighbor_AUCell"],
        ascending=[False, False, False], kind="stable",
    ).head(top_tfs).copy().reset_index(drop=True)
    candidates.insert(0, "tf_rank", np.arange(1, len(candidates) + 1))
    candidates.insert(0, "Regimen", regimen)
    candidates.insert(0, "query_patient", str(query_patient))
    # Preserve motif-pruned edges as *candidate targets*, not query-confirmed interactions.
    links = candidates[["query_patient", "Regimen", "tf_rank"] + keys].merge(
        edges, on=keys, how="left", validate="one_to_many",
    ).dropna(subset=["Target"])
    status = "DESCRIPTIVE_REGULON_EVIDENCE"
    if not valid.any():
        status = "DESCRIPTIVE_ONLY_INSUFFICIENT_R_NR_PER_REGULON"
    return candidates, links, {**base, "status": status}


def run(args):
    from .io import (
        load_mil, open_h5_matrix, read_obs_column, read_gene_names,
        patient_subsample, sha256_file, unit_vec,
    )
    from .predict import embedding_from_rows
    import h5py
    import torch

    if args.scale == "counts" and not (args.target_sum and args.target_sum > 0):
        raise ValueError("Raw-count input requires verified positive --target-sum")
    model, genes, regimens, cfg, device = load_mil(args.model_dir, args.device)
    ck_hash = sha256_file(Path(args.model_dir) / "final_augmented_MIL_159.pt")
    emb, hist, manifest = load_reference(args.atlas_dir, ck_hash, args.expected_patients)
    if emb.shape[1] != cfg["n_attention_heads"] * cfg["cell_embed_dim"]:
        raise ValueError("Reference embeddings are incompatible with the MIL architecture")
    if not set(hist.Regimen).issubset(set(regimens)):
        raise ValueError("Reference contains treatments absent from model vocabulary")
    auc, edges = load_regulons(args.atlas_dir, hist)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    neighbor_rows, tf_rows, target_rows, coverage_rows = [], [], [], []
    with h5py.File(args.input, "r") as f, torch.inference_mode():
        query_genes = read_gene_names(f)
        loc = query_genes.get_indexer(pd.Index(genes))
        coverage = (loc >= 0).mean()
        if coverage < args.min_gene_coverage:
            raise ValueError(f"Gene coverage {coverage:.1%} below threshold {args.min_gene_coverage:.1%}")
        matrix = open_h5_matrix(f, args.layer)
        if args.patient_col:
            query_ids = read_obs_column(f, args.patient_col)
        else:
            query_ids = np.repeat(args.patient_id or Path(args.input).stem, matrix.shape[0])
        for patient in sorted(set(query_ids)):
            if not patient or str(patient).lower() in ("nan", "none"):
                raise ValueError("Missing query patient ID")
            rows = np.flatnonzero(query_ids == patient)
            rows = patient_subsample(rows, args.max_cells, args.seed, patient, 0)
            # Query outputs are per actual candidate treatment, matching historical bank.
            h = embedding_from_rows(model, matrix, rows, loc, args.scale,
                                    args.target_sum, device, args.chunk_size)
            for ti, regimen in enumerate(regimens):
                _, _, z = model.patient_logit_from_h(h, ti, return_attention=True)
                q = unit_vec(z.detach().cpu().numpy())
                neighbors = cosine_neighbors(q, emb, hist, regimen, patient)
                # Exact R−NR Delta: include ALL eligible historical patients of
                # this regimen; never compute the score using only top-k neighbours.
                ref_r = neighbors.loc[neighbors.Response.eq("R"), "cosine_similarity"]
                ref_nr = neighbors.loc[neighbors.Response.eq("NR"), "cosine_similarity"]
                mean_r = float(ref_r.mean()) if len(ref_r) else np.nan
                mean_nr = float(ref_nr.mean()) if len(ref_nr) else np.nan
                exact_delta = mean_r - mean_nr if len(ref_r) and len(ref_nr) else np.nan
                display_n = neighbors.head(args.neighbor_top_k).copy()
                display_n.insert(0, "query_patient", patient)
                neighbor_rows.append(display_n)
                tf, targets, coverage_row = summarize_tf_candidates(
                    neighbors, auc, edges, regimen, patient,
                    neighbors_per_class=args.neighbors_per_class,
                    top_tfs=args.top_tfs,
                    min_group_n=args.min_group_n,
                )
                coverage_row["query_gene_coverage"] = float(coverage)
                coverage_row.update({
                    "Mean_R_similarity": mean_r,
                    "Mean_NR_similarity": mean_nr,
                    "Exact_RminusNR": exact_delta,
                })
                coverage_rows.append(coverage_row)
                if len(tf): tf_rows.append(tf)
                if len(targets): target_rows.append(targets)
                print(f"{patient} | {regimen} | refs {len(neighbors)} | {coverage_row['status']}", flush=True)
            del h
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Store all-regimen exact scores independently of nearest-neighbour TF evidence.
    score_df = pd.DataFrame(coverage_rows)
    score_df["Exact_rank"] = score_df.groupby("query_patient")["Exact_RminusNR"].rank(
        ascending=False, method="first"
    ).astype("Int64")
    score_columns = ["query_patient", "Regimen", "Mean_R_similarity",
                     "Mean_NR_similarity", "Exact_RminusNR", "Exact_rank",
                     "n_historical_R", "n_historical_NR"]
    score_df[score_columns].to_csv(out / "regimen_exact_scores.csv", index=False)
    print("Saved:", out / "regimen_exact_scores.csv")

    for name, frames in [
        ("nearest_historical_patients.csv", neighbor_rows),
        ("candidate_TF_regulons.csv", tf_rows),
        ("candidate_TF_target_edges.csv", target_rows),
        ("TF_evidence_coverage.csv", [score_df]),
    ]:
        pd.concat(frames, ignore_index=True).to_csv(out / name, index=False) if frames else pd.DataFrame().to_csv(out / name, index=False)
        print("Saved:", out / name)


def main():
    ap = argparse.ArgumentParser(description="Nearest historical same-regimen MIL cosine neighbours + TF/target evidence")
    ap.add_argument("--input", type=Path, required=True, help="Actual single-cell gene-expression H5AD")
    ap.add_argument("--atlas-dir", type=Path, required=True, help="Locally available private 156-patient atlas")
    ap.add_argument("--model-dir", type=Path, default=Path("models/mil_final159"))
    ap.add_argument("--patient-col", default=None)
    ap.add_argument("--patient-id", default=None)
    ap.add_argument("--layer", default="X")
    ap.add_argument("--scale", choices=["counts", "log1p"], required=True)
    ap.add_argument("--target-sum", type=float, default=None)
    ap.add_argument("--max-cells", type=int, default=1024)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-gene-coverage", type=float, default=0.60)
    ap.add_argument("--neighbor-top-k", type=int, default=10)
    ap.add_argument("--neighbors-per-class", type=int, default=5)
    ap.add_argument("--top-tfs", type=int, default=15)
    ap.add_argument("--min-group-n", type=int, default=2)
    ap.add_argument("--expected-patients", type=int, default=156)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/nearest_tf"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if min(args.neighbor_top_k, args.neighbors_per_class, args.top_tfs, args.min_group_n) < 1:
        ap.error("Top-K and minimum sample-size arguments must be positive")
    run(args)


if __name__ == "__main__":
    main()
