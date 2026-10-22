# NSCLC-Regimen-MIL

**Treatment-conditioned, four-head attention MIL for research-use-only patient-level response scoring.** The repository now supports two separate workflows: (A) scoring with the released **NSCLC Final159 pretrained model**, and (B) **training a new cancer-specific model** from labelled scRNA-seq patients, then scoring genuinely new patients with that model. Custom training is an experimental fit-and-save workflow; it does **not** reproduce the paper's nested cross-validation or establish clinical benefit.

> **Workflow A — pretrained NSCLC:** real single-cell H5AD → optional scANVI reference mapping → frozen Final159 MIL scoring. Bulk-only reconstruction requires a separate validated workflow.
>
> **Workflow B — any supported cancer cohort:** labelled patient-level scRNA-seq H5AD → `create_model()` → saved custom checkpoint + regimen-specific R/NR reference bank → unseen patient H5AD → `predict_custom()` → Exact R−NR regimen scores.
>
> **Important data constraint:** Standard bulk deconvolution estimates **cell-type proportions**, not individual cells or a cell × gene matrix. Neither scANVI nor this MIL implementation can use a proportions table directly. For bulk-only patients, a *separately validated, reference-guided cell-level reconstruction / pseudo-bag construction* step is required between deconvolution and scANVI/MIL. **That reconstruction and the deconvolution runner are not yet included in this repository.** The available CLI works today with genuine single-cell counts.

## New: train your own cancer-type model and score unseen patients

The optional `nsclc_regimen_mil.custom` module reuses the four-head gated-attention architecture but **trains a new checkpoint**. It does not reuse the pretrained NSCLC parameters or require ten regimens. Treatment vocabulary and gene order are learned from the **training cohort** and saved for future inference.

**Training input** is a real single-cell H5AD with raw integer-like UMI counts in `layers["counts"]` (or `X`) and three patient-consistent `obs` columns: `Patient`, `Treatment` (actual received regimen), and `Response` (`R`/`NR`; `response_binary` also supported). Each eligible regimen needs at least **two R and two NR independent patients**; this is a program guard, not a sample-size adequacy claim.

```bash
# 1. Train once using labelled patients
python -m nsclc_regimen_mil.custom create \
  --input /path/to/train.h5ad \
  --out-dir models/my_cancer \
  --patient-col Patient \
  --treatment-col Treatment \
  --response-col Response \
  --layer counts \
  --target-sum 10000 \
  --epochs 35 \
  --cells-per-bag 256 \
  --pool-cells 1024 \
  --device cpu

# 2. Score genuinely new patients; no response or treatment labels are needed
python -m nsclc_regimen_mil.custom score \
  --input /path/to/new_patients.h5ad \
  --model-dir models/my_cancer \
  --patient-col Patient \
  --layer counts \
  --max-cells 1024 \
  --repeats 3 \
  --output outputs/my_cancer_scores.csv \
  --device cpu
```

For a **single-patient** H5AD without an `obs["Patient"]` column, replace `--patient-col Patient` in the score command with `--patient-id NewPatient01`. Change `--layer counts` to `--layer X` when raw counts are in `X`. A CUDA-enabled PyTorch installation may use `--device cuda`.

**Saved model directory:** `custom_mil.pt`, `gene_order.csv`, `treatment_vocabulary.csv`, `reference_centroids.npz`, `reference_coverage.csv`, and `training_metadata.json`. **Keep these files together.** The checkpoint and reference bank are checked for consistency, input genes are automatically aligned, and query patient IDs found in training are rejected. The new patient inherits the saved normalization target; do **not** supply `--target-sum` when scoring.

**Output:** `outputs/my_cancer_scores.csv` contains one row per patient × trained regimen, with `Exact_RminusNR`, `Score_SD`, `Exact_rank`, `Mean_R_similarity`, `Mean_NR_similarity`, and separate `raw_R_score`. The score is the mean cosine similarity to historical regimen-matched **R** patients minus the corresponding mean for **NR** patients, not a calibrated response probability or causal treatment benefit. The companion `*_repeats.csv` stores per-repeat results.

**Test the module:** `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python test_custom_model.py`. The synthetic end-to-end test passed, and the training/score commands were also run successfully on a TNBC cohort (28 training patients, nine unseen test patients, three regimens, 15,989 genes; 27 score rows and 100% matched model genes). These are **software operation checks**, not evidence of independent clinical validity.

For full input requirements, new-patient examples, integrity checks, and limitations, see **[CUSTOM_MODEL_GUIDE.md](CUSTOM_MODEL_GUIDE.md)**. Do not commit patient-level H5ADs, locally trained checkpoints or `reference_coverage.csv` without data-sharing permission.

---

## 1. Which data do you have?

| Starting input | Stage 1: deconvolution | Stage 2: scANVI | Stage 3: Final159 MIL |
|---|---|---|---|
| **Real scRNA-seq**, raw UMI cell × gene counts | Skip | Run reference mapping | Run directly on the same single-cell H5AD |
| **Bulk RNA-seq only**, sample × gene expression | Required to estimate composition | **Not directly possible from proportions**; requires validated cell-level reconstruction | **Not directly possible from proportions**; requires compatible reconstructed cell-level input |
| **Bulk RNA-seq + independently measured matching scRNA-seq** | Optional | Run on the actual scRNA-seq data | Run on the actual scRNA-seq data |

These pathways must be reported separately: scores derived from *synthetic pseudo-bags* should not be described as measured single-cell predictions.

## 2. Install the inference toolkit

```bash
git clone https://github.com/babySherlock/NSCLC-Regimen-MIL.git
cd NSCLC-Regimen-MIL
git lfs install
git lfs pull
python -m pip install -r requirements.txt
python -m pip install -e .
```

For a server that only permits SSH over port 443, configure GitHub's SSH-over-443 endpoint and then clone through that remote instead. GPU is recommended for scANVI query adaptation, while the MIL command also accepts `--device cpu`.

## 3. Stage 1 — Bulk RNA-seq deconvolution

**Goal:** estimate the major immune/cell-type composition of each bulk RNA-seq patient using an appropriate annotated scRNA-seq reference.

**Required inputs**
- Bulk expression table (samples × genes), with gene identifiers and quantification units documented.
- An annotated scRNA-seq reference with compatible genes and biologically appropriate cell types.
- Patient/sample identifiers, treatment, and (if evaluating predictions) response labels kept separate from feature generation.

**Expected output:** a patient × cell-type fraction matrix, with nonnegative proportions and a documented normalization/reference procedure.

**Implementation status:** the uploaded repository does **not** currently contain a runnable bulk deconvolution script, a validated pseudo-cell expression constructor, or a prescribed deconvolution reference. Do not run either of the next two CLI tools directly on a cell-fraction CSV. Bulk-only end-to-end inference is **not yet supported**. The deconvolution method and cell-level conversion need to be packaged and validated before that can be advertised as executable.

Users who already have genuine raw-count single-cell H5AD input can start at Stage 2 without deconvolution.

## 4. Stage 2 — scANVI reference mapping

**Purpose:** map real query cells to the pretrained reference cell-type categories.

**Input:** an H5AD with one row per cell, raw nonnegative integer-like UMI counts (in `X` or a named layer), compatible gene identifiers, and optionally a patient ID column in `obs`.

```bash
python -m nsclc_regimen_mil.annotate_scanvi \
  --input /path/to/query_cells.h5ad \
  --patient-col Patient \
  --layer counts \
  --model-dir models/scanvi_reference \
  --epochs 30 \
  --output outputs/query_cell_annotations.csv
```

If the counts are stored in `adata.X`, replace `--layer counts` with `--layer X`. If the H5AD represents a single patient and has no patient column, omit `--patient-col`. Adaptation occurs on the query data, not by retraining the released reference checkpoint.

**Output:** a CSV with `barcode`, `predicted_celltype`, `scanvi_pmax`, `scanvi_margin`, and `scanvi_confident_label`. The latter flags uncertain cells as `Unknown` using the packaged thresholds.

**Current interface boundary:** this command produces an annotation CSV; it does **not** create a new annotated H5AD. The MIL command below does **not** read that CSV and does **not** currently require annotated cell types. Thus Stage 2 is an optional interpretability/quality-control stage for genuine single-cell input, not a mandatory transformation consumed by Stage 3. Making the two scripts an automatically connected pipeline is future integration work.

## 5. Stage 3 — Final159 patient-level MIL scoring

**Input:** the **cell × gene H5AD**, not the annotation CSV or bulk proportions. Genes are automatically aligned to the model's fixed 28,742-gene vocabulary; missing genes are zero-filled. Use the genuine query cell matrix, not the scANVI latent embedding.


### Gene alignment of new patient datasets (automatic, but verify first)

The Final159 MIL model uses a **fixed ordered vocabulary of 28,742 genes**, stored in `models/mil_final159/final_multihead_MIL_gene_order.csv` (column `gene`). A query H5AD **does not need to have its columns manually reordered**: `predict.py` and `nearest_tf.py` use `adata.var_names` (the H5AD `/var` index) to find each required gene, remap columns **into model order**, zero-fill missing model genes and omit query-only genes. The original query file is not overwritten, and cell/patient ordering is retained.

**Requirements and caveats:**
- Gene IDs must use the **same naming system** as the model vocabulary (e.g. HGNC symbols versus Ensembl IDs). This package does **not** perform automatic Ensembl-to-symbol conversion, species ortholog mapping, case correction, or alias resolution. Check ID systems before inferring. The CLI rejects duplicate query `var_names`; resolve duplicates or mapping collisions carefully, without arbitrarily discarding expression.
- **Missing model genes are filled with numerical zeros as input placeholders**; this is not evidence of true zero biological expression. Extra query genes are omitted from the final model feature matrix.
- Inspect the matched-gene count and `matched / 28,742` coverage. The CLI default `--min-gene-coverage 0.60` is only a *minimum execution guard*, **not a validated quality or predictive-accuracy threshold**. A higher user-defined threshold (e.g. `0.90`) may be used to flag incomplete data, but no cutoff guarantees transferable accuracy.
- **Expression preprocessing is a separate requirement from gene ordering.** In `--scale counts` mode, the implemented code normalizes each cell across **all genes present in the query counts matrix** to the user-specified, **verified training-compatible** `--target-sum`, then applies `log1p` and aligns to the model gene order. Do not first truncate the counts matrix to model genes unless that matches the validated training protocol. In `--scale log1p` mode, input values must already be normalized compatibly; the code does not re-normalize them. An audit of 256 cells from the original NSCLC H5AD was consistent with a pre-log1p normalization target of **3,565**. This does not establish cross-cohort preprocessing equivalence; verify input library and gene-space semantics.

**Read-only gene-coverage audit (before prediction):**

```bash
cd NSCLC-Regimen-MIL
python - <<'PY'
import pandas as pd
import anndata as ad

query_path = "/path/to/new_patient.h5ad"  # change this
reference_genes = pd.Index(
    pd.read_csv("models/mil_final159/final_multihead_MIL_gene_order.csv")["gene"].astype(str)
)
adata = ad.read_h5ad(query_path, backed="r")
try:
    query_genes = pd.Index(adata.var_names.astype(str))
    if not query_genes.is_unique:
        raise ValueError("Duplicate query gene IDs: inspect /var before prediction")
    indexer = query_genes.get_indexer(reference_genes)
    n_matched = int((indexer >= 0).sum())
    print(f"Cells: {adata.n_obs} | Query genes: {len(query_genes)}")
    print(f"Model genes matched: {n_matched}/{len(reference_genes)} ({n_matched / len(reference_genes):.2%})")
    print("Already in identical order:", query_genes.equals(reference_genes))
    print("First missing model genes:", reference_genes[indexer < 0][:20].tolist())
    print("First model gene IDs:", reference_genes[:10].tolist())
    print("First query gene IDs:", query_genes[:10].tolist())
    print("Layers available:", list(adata.layers.keys()))
finally:
    if getattr(adata, "isbacked", False):
        adata.file.close()
PY
```

If gene coverage is unexpectedly low, investigate gene identifiers, reference assembly and preprocessing **before** running the model. For example, with an already training-compatible `log1p` H5AD, use `--layer X --scale log1p --min-gene-coverage 0.90` (the `0.90` here is a conservative **user-selected** gate, not a validated threshold). The same automatic alignment and coverage checks apply to historical TF retrieval through `nearest_tf.py`.


```bash
python -m nsclc_regimen_mil.predict \
  --input /path/to/query_cells.h5ad \
  --patient-col Patient \
  --layer counts \
  --scale counts \
  --target-sum 3565 \
  --model-dir models/mil_final159 \
  --reference-centroids models/mil_final159/reference_centroids.npz \
  --max-cells 1536 \
  --repeats 3 \
  --output outputs/regimen_scores.csv
```

For the released Final159 NSCLC checkpoint, a 256-cell audit of its source training H5AD supports the **3,565** pre-log1p target. Use this example only when query raw-UMI preprocessing and library-size semantics match the reference. The separate **custom cancer-type trainer** uses its own configurable target (e.g. 10,000), saved with each new model and reused automatically at scoring time.

If the matrix already contains log1p-normalized expression *matching the model's original preprocessing*, use `--scale log1p --layer X` and omit `--target-sum`. For a one-patient file without `obs['Patient']`, omit `--patient-col` and use `--patient-id Patient01`.

**Outputs:**
- `outputs/regimen_scores.csv` — one row per patient × regimen (10 regimens), including `raw_R_score` and `raw_R_rank`.
- With reference centroids: `Mean_R_similarity`, `Mean_NR_similarity`, `Exact_RminusNR`, `Exact_rank`, and repeat variability.
- `outputs/regimen_scores_repeats.csv` — one row per patient × regimen × subsample repeat.

**Score interpretation:** `raw_R_score` is the model's sigmoid response output under a specified regimen, not a validated clinical probability. `Exact_RminusNR` is a **cosine similarity difference** to regimen-matched historical R and NR patient embeddings, **not** a response probability. These are distinct outputs and their rankings need not agree. For a new patient, counterfactual rankings across regimens are exploratory; the original observational cohort does not identify unreceived treatment outcomes.

## 6. Packaged resources

| Path | Role |
|---|---|
| `models/mil_final159/final_augmented_MIL_159.pt` | Frozen MIL checkpoint |
| `models/mil_final159/final_multihead_MIL_gene_order.csv` | Exact 28,742-gene input order |
| `models/mil_final159/treatment_vocabulary.csv` | Ten regimen names and indices |
| `models/mil_final159/reference_centroids.npz` | Historical regimen-specific R/NR embedding centroids for optional similarity scoring |
| `models/mil_final159/final_model_calibration_threshold.json` | Historical calibration/threshold metadata; not automatically applied in the CLI |
| `models/scanvi_reference/` | Pretrained scANVI reference model |
| `src/nsclc_regimen_mil/annotate_scanvi.py` | Single-cell reference mapping CLI |
| `src/nsclc_regimen_mil/predict.py` | Patient-level ten-regimen scoring CLI |
| `src/nsclc_regimen_mil/custom.py` | Custom cancer-type model training and new-patient scoring (`create` / `score`) |
| `CUSTOM_MODEL_GUIDE.md` | Custom training/scoring instructions and output interpretation |
| `test_custom_model.py` | Synthetic end-to-end custom model checks |

## 7. Limitations, privacy, and citation

This is a **research-only**, observational-data-derived pipeline, not an approved diagnostic tool or a basis for prescribing treatment. Cross-platform effects, cell selection, cell-type composition, unknown gene identifiers, batch structure, normalization differences, and synthetic-cell assumptions can change the output. The released code does not establish transfer performance on unseen bulk-only patients.

Only distribute pretrained checkpoints, derived centroids, reference registries, and example data when their source licenses and privacy implications permit release. The repository does not include original patient-level expression matrices. Add the applicable paper, accession, model attribution, and a chosen license after checking permissions.

## Historical cosine similarity and TF-target evidence (optional)

After cell-type annotation and Final159 scoring, users with access to an **authorized private reference atlas** can additionally retrieve nearest historical patients and subtype-specific TF-target regulon evidence using the `nearest_tf` command. The reference atlas is built **once** from the 156-patient patientwise AUCell/MIL notebook; it is not created or retrained during query inference.

```bash
python -m nsclc_regimen_mil.nearest_tf \
  --input /path/to/your_singlecell.h5ad \
  --patient-id Query01 \
  --model-dir models/mil_final159 \
  --atlas-dir /path/to/authorized/Private_Atlas_156_PatientWise_v1 \
  --layer X --scale log1p \
  --max-cells 1024 \
  --output-dir outputs/nearest_tf
```

`--scale log1p` is valid **only** when your query data have the training-compatible preprocessing. With raw counts, use `--scale counts --target-sum YOUR_VERIFIED_TRAINING_TARGET`. For multiple patients, use `--patient-col YOUR_OBS_COLUMN` instead of `--patient-id`.

The command additionally produces `regimen_exact_scores.csv`, calculated from **all** eligible regimen-specific historical R and NR patient embeddings, not the displayed top-k neighbours. It also produces `nearest_historical_patients.csv`, `candidate_TF_regulons.csv`, `candidate_TF_target_edges.csv`, and `TF_evidence_coverage.csv`. Candidate TFs and targets are retrieved **historical associations**, not experimentally confirmed drivers for the query patient. Similarities are restricted to reference patients who actually received each candidate regimen, and the reference does not establish causal benefit from alternate therapies.

For details, see [REFERENCE_ATLAS_GUIDE.md](REFERENCE_ATLAS_GUIDE.md). The 156 patient-derived records are **not part of this public repository** until redistribution permissions and privacy review explicitly permit release. These records may be kept in a separate private repository or provided under controlled access.

The reference bank lives in a separately controlled Private repository (`babySherlock/NSCLC-Regimen-MIL-PrivateAtlas`). Access to the public CLI does not grant access to the historical patient-level bank. Use only data that your institution permits uploading or sharing.
