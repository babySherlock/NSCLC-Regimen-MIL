"""CLI: new single-cell patient(s) -> all ten Final159 regimen scores.

Exact_RminusNR is a cosine similarity difference, NOT a response probability or causal benefit.
"""
import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from .io import (
    load_mil, open_h5_matrix, read_obs_column, read_gene_names,
    get_csr, csr_to_torch, align_and_normalize, patient_subsample,
    unit_vec, sha256_file,
)


@torch.inference_mode()
def embedding_from_rows(model, source, rows, model_to_query, scale, target_sum, device, chunk_size=256):
    blocks = []
    for begin in range(0, len(rows), chunk_size):
        c = get_csr(source, rows[begin:begin + chunk_size])
        aligned = align_and_normalize(c, model_to_query, scale, target_sum)
        blocks.append(model.encode_cells(csr_to_torch(aligned, device)))
    if not blocks:
        raise ValueError("Patient has zero cells")
    return torch.cat(blocks, dim=0)


def load_centroids(path, regimen_order, embedding_dim, ck_hash):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as archive:
        if archive["regimens"].astype(str).tolist() != regimen_order:
            raise ValueError("Reference centroid regimen vocabulary mismatch")
        if "checkpoint_sha256" not in archive.files or str(archive["checkpoint_sha256"].item()) != ck_hash:
            raise ValueError("Reference centroids were exported from a different checkpoint")
        R = np.asarray(archive["R"], dtype=np.float64)
        NR = np.asarray(archive["NR"], dtype=np.float64)
        if R.shape != (len(regimen_order), embedding_dim) or NR.shape != R.shape:
            raise ValueError("Reference centroids have incorrect shape")
        if not (np.isfinite(R).all() and np.isfinite(NR).all()):
            raise ValueError("Reference centroids contain non-finite values")
        return R, NR


def run_inference(args):
    model_dir = Path(args.model_dir)
    model, genes, regimens, cfg, device = load_mil(model_dir, args.device)
    ck_hash = sha256_file(model_dir / "final_augmented_MIL_159.pt")
    embedding_dim = cfg["n_attention_heads"] * cfg["cell_embed_dim"]
    bank = load_centroids(args.reference_centroids, regimens, embedding_dim, ck_hash)

    if args.scale == "counts" and (args.target_sum is None or args.target_sum <= 0):
        raise ValueError("Counts input requires --target-sum matching training preprocessing")
    if args.repeats <= 0 or args.max_cells <= 0:
        raise ValueError("--repeats and --max-cells must be positive")

    raw_rows = []
    with h5py.File(args.input, "r") as f, torch.inference_mode():
        query_genes = read_gene_names(f)
        model_to_query = query_genes.get_indexer(pd.Index(genes))
        matched = int((model_to_query >= 0).sum())
        coverage = matched / len(genes)
        print(f"Query genes: {len(query_genes)} | MIL common: {matched}/{len(genes)} ({coverage:.2%})")
        if coverage < args.min_gene_coverage:
            raise ValueError("Too few genes match the MIL vocabulary; check gene IDs")
        if coverage < 0.8:
            print("WARNING: incomplete gene coverage; resulting treatment rankings are exploratory")

        matrix = open_h5_matrix(f, args.layer)
        if args.patient_col:
            patient_ids = read_obs_column(f, args.patient_col)
        else:
            patient_ids = np.full(matrix.shape[0], args.patient_id or Path(args.input).stem, dtype=object)
        if len(patient_ids) != matrix.shape[0]:
            raise ValueError("Patient IDs and matrix row count mismatch")
        found_patients = sorted(set(patient_ids))
        if args.only_patients:
            absent = set(args.only_patients) - set(found_patients)
            if absent:
                raise ValueError(f"Requested patients missing from input: {sorted(absent)}")
            found_patients = [p for p in found_patients if p in set(args.only_patients)]
        for patient in found_patients:
            if not patient or patient.lower() in ("nan", "none"):
                raise ValueError("Missing patient identifier")
            rows = np.flatnonzero(patient_ids == patient)
            for repeat in range(args.repeats):
                chosen = patient_subsample(rows, args.max_cells, args.seed, patient, repeat)
                h = embedding_from_rows(
                    model, matrix, chosen, model_to_query, args.scale,
                    args.target_sum, device, args.chunk_size,
                )
                for t_index, regimen in enumerate(regimens):
                    logit, _, z_heads = model.patient_logit_from_h(h, t_index, return_attention=True)
                    record = {
                        "Patient": patient, "repeat": repeat, "Regimen": regimen,
                        "N_cells_used": int(len(chosen)),
                        "raw_R_score": float(torch.sigmoid(logit).item()),
                    }
                    if bank is not None:
                        q = unit_vec(z_heads.detach().cpu().numpy())
                        mean_r = float(q @ bank[0][t_index])
                        mean_nr = float(q @ bank[1][t_index])
                        record.update(
                            Mean_R_similarity=mean_r,
                            Mean_NR_similarity=mean_nr,
                            Exact_RminusNR=mean_r - mean_nr,
                        )
                    raw_rows.append(record)
                print(f"{patient}: repeat {repeat+1}/{args.repeats} done", flush=True)

    repeats_df = pd.DataFrame(raw_rows)
    group = ["Patient", "Regimen"]
    summary = repeats_df.groupby(group, as_index=False).agg(
        raw_R_score=("raw_R_score", "mean"),
        raw_R_score_SD=("raw_R_score", "std"),
        N_cells_used=("N_cells_used", "first"),
    )
    summary["raw_R_score_SD"] = summary["raw_R_score_SD"].fillna(0)
    summary["raw_R_rank"] = summary.groupby("Patient")["raw_R_score"].rank(
        ascending=False, method="first"
    ).astype(int)
    if bank is not None:
        extra = repeats_df.groupby(group, as_index=False).agg(
            Mean_R_similarity=("Mean_R_similarity", "mean"),
            Mean_NR_similarity=("Mean_NR_similarity", "mean"),
            Exact_RminusNR=("Exact_RminusNR", "mean"),
            Score_SD=("Exact_RminusNR", "std"),
        )
        summary = summary.merge(extra, on=group, validate="one_to_one")
        summary["Score_SD"] = summary["Score_SD"].fillna(0)
        summary["Exact_rank"] = summary.groupby("Patient")["Exact_RminusNR"].rank(
            ascending=False, method="first"
        ).astype(int)
        summary = summary.sort_values(["Patient", "Exact_rank"])
    else:
        summary = summary.sort_values(["Patient", "raw_R_rank"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False, encoding="utf-8-sig")
    repeats_df.to_csv(out.with_name(out.stem + "_repeats.csv"), index=False, encoding="utf-8-sig")
    print(f"Saved {len(summary)} rows to {out}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Final159 exploratory ten-regimen MIL scoring")
    parser.add_argument("--input", type=Path, required=True, help="Input scRNA-seq H5AD")
    parser.add_argument("--model-dir", type=Path, default=Path("models/mil_final159"))
    parser.add_argument("--patient-col", default=None, help="obs column grouping cells by patient")
    parser.add_argument("--patient-id", default=None, help="Patient name for single-patient files")
    parser.add_argument("--only-patients", nargs="+", default=None, help="Optional subset of patient IDs")
    parser.add_argument("--layer", default="X")
    parser.add_argument("--scale", choices=["counts", "log1p"], required=True)
    parser.add_argument("--target-sum", type=float, default=None)
    parser.add_argument("--max-cells", type=int, default=1536)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-gene-coverage", type=float, default=0.60)
    parser.add_argument("--reference-centroids", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/regimen_scores.csv"))
    parser.add_argument("--device", default=None)
    run_inference(parser.parse_args())

if __name__ == "__main__":
    main()
