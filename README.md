# NSCLC-Regimen-MIL

**A user-oriented, three-stage workflow for exploring treatment-response signals in new patient transcriptomic data.** This repository distributes pretrained Final159 treatment-conditioned, four-head Attention MIL weights and a separate scANVI reference mapping model. It is intended for **inference, not reproduction of model training or paper benchmarks**.

> **Workflow:** Bulk RNA-seq deconvolution → scANVI reference mapping → patient-level MIL scoring.
>
> **Important data constraint:** Standard bulk deconvolution estimates **cell-type proportions**, not individual cells or a cell × gene matrix. Neither scANVI nor this MIL implementation can use a proportions table directly. For bulk-only patients, a *separately validated, reference-guided cell-level reconstruction / pseudo-bag construction* step is required between deconvolution and scANVI/MIL. **That reconstruction and the deconvolution runner are not yet included in this repository.** The available CLI works today with genuine single-cell counts.

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

```bash
python -m nsclc_regimen_mil.predict \
  --input /path/to/query_cells.h5ad \
  --patient-col Patient \
  --layer counts \
  --scale counts \
  --target-sum VERIFIED_TRAINING_TARGET_SUM \
  --model-dir models/mil_final159 \
  --reference-centroids models/mil_final159/reference_centroids.npz \
  --max-cells 1536 \
  --repeats 3 \
  --output outputs/regimen_scores.csv
```

Replace `VERIFIED_TRAINING_TARGET_SUM` with a **numerical library-size target verified against the training preprocessing**. The released code intentionally requires this value for raw counts. A prior external demonstration used 3565 as an **unverified approximation**, which should not be presented as the established training setting.

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

## 7. Limitations, privacy, and citation

This is a **research-only**, observational-data-derived pipeline, not an approved diagnostic tool or a basis for prescribing treatment. Cross-platform effects, cell selection, cell-type composition, unknown gene identifiers, batch structure, normalization differences, and synthetic-cell assumptions can change the output. The released code does not establish transfer performance on unseen bulk-only patients.

Only distribute pretrained checkpoints, derived centroids, reference registries, and example data when their source licenses and privacy implications permit release. The repository does not include original patient-level expression matrices. Add the applicable paper, accession, model attribution, and a chosen license after checking permissions.
