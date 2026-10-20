# 156-patient cosine-similarity + TF/target reference atlas

This module uses the **already-finished**, patient-by-patient local computation. It does not rerun GRNBoost2, cisTarget, AUCell, MIL training, or scANVI.

## 1. Prepare the completed 156-patient private atlas

In `NSCLC_156Patients_SCENIC_AUCell_MIL_逐患者保存断点续跑_FIX156_v3_anndata兼容.ipynb`, run the **section 5: 汇总全部已完成患者** cell **once**. Its input is the saved files under:

`/data/gj/project/SCVI/Private_Atlas_156_PatientWise_v1`

Verify that it prints `status: complete` and saves the following files:

```text
Private_Atlas_156_PatientWise_v1/
    reference_patient_embeddings.npz       # 156 × 512, actual-treatment embeddings + patient IDs, R/NR
    reference_patient_regulons.csv.gz       # historical patient × subtype × TF/regulon AUCell
    reference_regulon_targets.csv.gz        # shared motif-pruned subtype TF → target edges
    atlas_manifest_patientwise.json         # completed run, model SHA256 and settings
```

The completed atlas reported 156 patients, 10 regimens, 286,256 historical AUCell records and 156 × 512 MIL embeddings. The private atlas is uploaded separately to `babySherlock/NSCLC-Regimen-MIL-PrivateAtlas` (access controlled). An atlas is not accepted if it is partial, has the wrong checkpoint SHA256, contains repeated patient IDs, or has mismatching patient outcome/treatment metadata. These are derived **patient-level** data; see section 4 before publishing.

## 2. Run the new nearest-patient cosine lookup

Install as in the main README and provide genuine scRNA-seq counts (or a fully validated and appropriately normalized cell-level input).

Example using a log1p normalized H5AD **only if normalization matches model training**:

```bash
python -m nsclc_regimen_mil.nearest_tf \
  --input /path/to/new_patient.h5ad \
  --patient-id Query01 \
  --layer X --scale log1p \
  --model-dir models/mil_final159 \
  --atlas-dir /data/gj/project/SCVI/Private_Atlas_156_PatientWise_v1 \
  --max-cells 1024 --chunk-size 128 \
  --neighbor-top-k 10 --neighbors-per-class 5 \
  --top-tfs 15 \
  --output-dir outputs/nearest_tf
```

For genuinely raw integer counts, use `--scale counts --target-sum <VERIFIED_TRAINING_VALUE>` and optionally `--layer counts`. Do **not** use the previously mentioned 3565 as a certified normalization target; that value was an unverified exploratory estimate.

For multi-patient H5AD input, replace `--patient-id Query01` with `--patient-col Patient` (use your actual `adata.obs` column name).

Outputs:

| File | Meaning |
|---|---|
| `nearest_historical_patients.csv` | Historical patients who **actually received the candidate regimen**, with cosine similarities; contains true R/NR, ranks and query IDs |
| `regimen_exact_scores.csv` | Mean cosine to **all** same-regimen historical R and NR, their Exact R−NR Delta, and 10-regimen ranking (missing group → NaN) |
| `candidate_TF_regulons.csv` | TF/regulon AUCell support in nearest historical responders, plus descriptive same-regimen R−NR contrast where enough samples exist |
| `candidate_TF_target_edges.csv` | Motif-supported subtype TF→target edges for the retrieved candidate regulons |
| `TF_evidence_coverage.csv` | Counts of historical R/NR and whether interpretable TF evidence is available for each regimen |

## 3. Scientific interpretation

- The query embedding is generated using a **candidate** treatment embedding; each historical embedding was generated using its patient's **actually received** treatment. Neighbours are only compared **within that same actual regimen**.
- `cosine_similarity` measures embedding proximity, not response probability or causal treatment benefit.
- A retrieved TF/target pair is **historical regulon evidence**; it has not been experimentally demonstrated to function in the query patient.
- The original `reference_centroids.npz` provides historical R/NR reference centroids; its validity for **Exact R−NR Delta** depends on those centroids being arithmetic means of individually unit-normalized reference embeddings, without renormalizing each mean. This lookup CLI instead calculates Exact R−NR Delta directly from the complete 156-patient reference archive. The original centroid file does not contain the individual embeddings required for neighbour retrieval.
- `historical_R_minus_NR_AUCell` is purely descriptive. It is `NaN` unless the historical regimen has at least 2 measured R and 2 measured NR for that subtype/regulon. It is not a p-value.
- Query samples that overlap the 156 reference IDs are excluded from their own nearest-neighbour list, but a retrospective performance assessment would require proper patient-wise held-out evaluation beyond self-exclusion.
- Most regimens are represented by small cohorts. Absence of a comparator group makes R/NR contrasts unavailable and the status CSV will indicate this.
- If the input is bulk RNA-seq, do **not** feed the cell-proportion output of deconvolution into this CLI. You need a separately validated cell-level representation, with explicit disclosure that it is synthetic.

## 4. Publishing the reference data

The GitHub repository `babySherlock/NSCLC-Regimen-MIL` is presently **public**. The atlas contains **individual patient-derived embeddings, linked outcomes/regimens and AUCell profiles**. Do not run `git add` or `git push` for those values on that public repo until the dataset license and any permitted patient-level derivative distribution have been reviewed and authorization is documented. Data can remain in the local `Private_Atlas_156_PatientWise_v1` directory (the CLI accepts its path), or be stored in a **separate private GitHub repository** with controlled access; Git LFS is not required for the current small atlas files. A private branch in a public repository **is not private**.

The code and schema documentation can be published publicly without disclosing the 156 patient-level atlas. Only include the atlas in a public release if you have established redistribution rights and an appropriate privacy assessment.
