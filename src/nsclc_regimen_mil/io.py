"""I/O and model-loading helpers for the frozen Final159 MIL checkpoint."""
from pathlib import Path
import hashlib
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from anndata.io import read_elem, sparse_dataset
from .model import TreatmentConditionedMultiHeadAttentionMIL

MODEL_ARG_KEYS = (
    "n_genes", "n_treatments", "n_attention_heads", "cell_embed_dim",
    "treatment_embed_dim", "attention_dim", "classifier_hidden",
    "cell_dropout", "patient_dropout",
)


def sha256_file(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mil(model_dir, device=None):
    model_dir = Path(model_dir)
    ckfile = model_dir / "final_augmented_MIL_159.pt"
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(ckfile, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    genes = pd.read_csv(model_dir / "final_multihead_MIL_gene_order.csv")["gene"].astype(str).tolist()
    vocab = pd.read_csv(model_dir / "treatment_vocabulary.csv").sort_values("treatment_index")
    regimens = vocab["Treatment"].astype(str).tolist()

    if len(genes) != cfg["n_genes"] or len(set(genes)) != len(genes):
        raise ValueError("Invalid checkpoint gene order / duplicate genes")
    if len(regimens) != cfg["n_treatments"] or regimens != list(ck["regimens"]):
        raise ValueError("Checkpoint regimen vocabulary mismatch")
    if vocab["treatment_index"].astype(int).tolist() != list(range(len(regimens))):
        raise ValueError("Treatment indices are not 0..9")
    model = TreatmentConditionedMultiHeadAttentionMIL(
        **{key: cfg[key] for key in MODEL_ARG_KEYS}
    ).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    return model, genes, regimens, cfg, device


def open_h5_matrix(h5file, layer="X"):
    """Open .X or a counts layer without parsing potentially malformed /uns."""
    node = h5file["X"] if layer == "X" else h5file["layers"][layer]
    return sparse_dataset(node) if isinstance(node, h5py.Group) else node


def read_obs_column(h5file, name):
    if name not in h5file["obs"]:
        raise KeyError(f"Missing obs[{name}]")
    return pd.Series(read_elem(h5file["obs"][name])).astype(str).str.strip().to_numpy()


def read_gene_names(h5file):
    genes = pd.Index(read_elem(h5file["var"]).index.astype(str))
    if not genes.is_unique:
        raise ValueError("Query contains duplicated gene names")
    return genes


def get_csr(source, rows):
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size and (np.diff(rows) <= 0).any():
        raise ValueError("Rows must be sorted and distinct")
    matrix = source[rows, :]
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(np.asarray(matrix, dtype=np.float32))
    matrix = matrix.tocsr().astype(np.float32)
    if matrix.data.size and (
        not np.isfinite(matrix.data).all() or (matrix.data < 0).any()
    ):
        raise ValueError("Matrix contains negative, NaN or infinite expression")
    return matrix


def csr_to_torch(csr, device):
    x = csr.tocoo()
    indices = torch.as_tensor(
        np.stack([x.row, x.col]), dtype=torch.int64, device=device
    )
    values = torch.as_tensor(x.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, x.shape, device=device).coalesce()


def align_and_normalize(csr, model_to_query, scale, target_sum=None):
    """Normalize first on all observed query genes, then zero-fill missing MIL genes."""
    if scale == "counts":
        if target_sum is None or target_sum <= 0:
            raise ValueError("Counts input requires positive --target-sum")
        if not np.allclose(csr.data, np.rint(csr.data), atol=1e-5):
            raise ValueError("Counts mode requires integer-like raw UMI counts")
        total = np.asarray(csr.sum(axis=1)).ravel()
        if (total <= 0).any():
            raise ValueError("Zero-library cells found; remove them before prediction")
        factors = target_sum / total
        csr = csr.multiply(factors[:, None]).tocsr()
        csr.data = np.log1p(csr.data).astype(np.float32)
    elif scale == "log1p":
        # Nonnegative log1p normalized input must already match reference preprocessing.
        if csr.nnz and csr.data.max() > 25:
            raise ValueError("log1p input has values >25; inspect normalization")
    else:
        raise ValueError("Scale must be counts or log1p")

    model_cols = np.flatnonzero(model_to_query >= 0)
    query_cols = model_to_query[model_cols]
    chunk = csr[:, query_cols].tocoo()
    aligned = sp.coo_matrix(
        (chunk.data, (chunk.row, model_cols[chunk.col])),
        shape=(csr.shape[0], len(model_to_query)), dtype=np.float32,
    ).tocsr()
    return aligned


def patient_subsample(rows, max_cells, seed, patient, repetition):
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) <= max_cells:
        return rows
    key = f"{seed}|{patient}|{repetition}"
    intseed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(intseed)
    return np.sort(rng.choice(rows, size=max_cells, replace=False))


def unit_vec(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Invalid zero-norm patient embedding")
    return v / norm
