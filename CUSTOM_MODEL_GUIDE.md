# Custom cancer-type MIL: train once, score new patients (experimental)

This optional module adds a **new cancer-specific model**; it **does not overwrite** the pretrained NSCLC Final159 model. The exported reference bank is made from *the actual treated responder/non-responder patients in the custom training set*, using the exact newly-trained checkpoint.

## 0. Install from GitHub

The custom module is included in the public repository at `src/nsclc_regimen_mil/custom.py`. You no longer need a separate ZIP patch. In a fresh checkout, clone the repository; in an existing checkout, pull the latest `main` branch:

```bash
git clone https://github.com/babySherlock/NSCLC-Regimen-MIL.git
cd NSCLC-Regimen-MIL
# Or, if already cloned: git pull --ff-only
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m nsclc_regimen_mil.custom --help
```

PyTorch, h5py, NumPy, pandas, and SciPy must be installed. Use the Python environment in which the existing repository is installed. The custom module does not import AnnData for the inference computation.

## 1. Your training H5AD, once per cancer type

Input must contain *raw integer UMI counts* (CSR sparse or dense) in `layers['counts']`, or `X` if `--layer X`. The cell `var_names` are gene IDs; no duplicated IDs. `obs` has three patient-consistent columns:

| Cell's `obs` column | Example | Required for training? |
|---|---|---|
| `Patient` | P001 | Yes |
| `Treatment` | DrugA+ChemoB | Yes; one **actually received** regimen per patient |
| `response_binary` | R or NR, alternatively 1 or 0 | Yes; observed patient outcome |

A minimum guard in this experimental implementation requires >=8 labelled patients total, >=2 R and >=2 NR patients **per treatment**, but this is merely an executable input constraint, not sufficient sample size for research validity.

```bash
python -m nsclc_regimen_mil.custom create \
  --input /path/to/train.h5ad \
  --out-dir models/my_cancer \
  --patient-col Patient \
  --treatment-col Treatment \
  --response-col response_binary \
  --layer counts \
  --target-sum 10000 \
  --epochs 35 \
  --cells-per-bag 256 \
  --pool-cells 1024 \
  --device cuda
```

Use `--device cpu` when CUDA is unavailable. The `--target-sum 10000` is chosen for the **newly trained model** and is saved automatically; new patients are normalized to the **saved training target**, so the user must **not** supply `--target-sum` to the custom scoring command. The existing NSCLC Final159 model used a different preprocessing target, audited as 3565.

The model directory will contain `custom_mil.pt`, `gene_order.csv`, `treatment_vocabulary.csv`, `reference_centroids.npz`, `reference_coverage.csv`, and `training_metadata.json`. Retain these files together. In particular, `reference_centroids.npz` is bound to its matching model checkpoint and may not be mixed with a different training run. `reference_coverage.csv` lists training patient IDs, so treat the directory as potentially sensitive and do not redistribute it without permission.

## 2. How to input **new patients** after the model is trained

**New-patient H5AD does not need Treatment or response_binary labels.** It needs only real scRNA-seq **raw UMI counts**, gene IDs, and patient IDs if multiple patients are in one file. The model cannot score treatments that were absent from its training set.

### Case A: one H5AD has multiple unseen patients

`new_patients.h5ad` has `obs['Patient']` grouping cells and raw UMI counts in `layers['counts']`:

```bash
python -m nsclc_regimen_mil.custom score \
  --input /path/to/new_patients.h5ad \
  --model-dir models/my_cancer \
  --patient-col Patient \
  --layer counts \
  --max-cells 1024 \
  --repeats 3 \
  --output outputs/my_cancer_scores.csv \
  --device cuda
```

### Case B: one H5AD contains only one unseen patient and lacks `obs['Patient']`

```bash
python -m nsclc_regimen_mil.custom score \
  --input /path/to/one_new_patient.h5ad \
  --model-dir models/my_cancer \
  --patient-id Query001 \
  --layer counts \
  --max-cells 1024 \
  --repeats 3 \
  --output outputs/Query001_scores.csv \
  --device cuda
```

If the raw counts are in `adata.X`, use `--layer X`. If the new one-patient file already has `obs['Patient']`, **use `--patient-col Patient`** instead of `--patient-id`.

The score command automatically performs the SAME saved target-sum + log1p normalization as training, aligns query gene columns to the saved `gene_order.csv`, fills missing genes with zeros, and calculates one response contrast for **each trained treatment**. A minimum gene coverage guard is `0.8` by default (not a validated biological accuracy threshold). Inspect gene identifiers and coverage. Additional query-only genes participate in the query library-size normalization before model alignment; large differences in gene panels can cause preprocessing shift even when coverage passes.

### Output

- `outputs/my_cancer_scores.csv`: each patient × trained regimen has `Exact_RminusNR`, `Score_SD`, `Exact_rank`, `Mean_R_similarity`, `Mean_NR_similarity`, `raw_R_score`, historical R/NR reference counts.
- `outputs/my_cancer_scores_repeats.csv`: scores for all individual repeated bag subsamples.

The final exploratory score is:

\[ S_{p,r}=\overline{\cos(q_{p,r},z_{i\in R_r})}-\overline{\cos(q_{p,r},z_{j\in NR_r})}\]

Cosines are computed with **individually L2-normalized** patient embeddings; means of historical vectors are *not* L2-renormalized, so the centroid dot products exactly reproduce averages of individual cosine similarities. `raw_R_score` is the neural classifier's separate sigmoid output. **These are not validated response probabilities or causal treatment benefits.**

`--repeats 3` produces sampling variability when a patient has more than `--max-cells` cells. If all cells are used in every repeat, the SD may be zero; no independent biological replicates are implied.

## 3. Safety and integrity checks

- Training patients must not be reused as supposedly unseen queries. The code rejects a query if its patient ID matches an ID in the training reference coverage file. Use unique IDs across training and new-patient datasets; distinct IDs do not rule out the same biological donor.
- The checkpoint, gene-order file, and reference bank are integrity checked before scoring.
- Training requires patient-level labels; scoring **does not** require labels or known received regimen for the new patient.
- An input of bulk RNA-seq, TPM table, deconvolved cell fractions, or log1p expression incorrectly marked as counts cannot directly run through this raw-UMI interface. No cell reconstruction module is included.
- The trainer is an **experimental fit-and-save implementation**, not a reproduction of the paper's nested cross-fitting workflow. We deliberately **did not add `evaluate_model()`**.

## 4. Smoke / end-to-end testing

After extracting the patch in your repo root, use:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python test_custom_model.py
```

This test creates temporary **synthetic training and unseen-query H5ADs**, calls both the Python API and CLI, and verifies model reload, 3 repeats, output CSV, mean/SD, gene reordering invariance, missing/extra gene handling, single- and multi-patient inputs, explicit reference cosine identity, train/query overlap guard, mismatched checkpoint/reference/gene files, and invalid labels/counts. The script removes synthetic temporary files afterwards. The test does **not** validate accuracy on a real cancer cohort.
