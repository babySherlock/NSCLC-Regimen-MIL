# NSCLC-Regimen-MIL

Research-use-only toolkit with the frozen **Final159** four-head treatment-conditioned Attention MIL checkpoint and a separate scANVI reference model for cell-subtype annotation.

## Assets

- `models/mil_final159/final_augmented_MIL_159.pt` — Final159 MIL checkpoint (`state_dict`, `config`)
- `models/mil_final159/final_multihead_MIL_gene_order.csv` — exact 28,742 input genes
- `models/mil_final159/treatment_vocabulary.csv` — fixed 10-regimen vocabulary
- `models/mil_final159/reference_centroids.npz` — optional per-regimen mean unit embeddings for original 156 historical R/NR patients; redistribution requires authorization
- `models/scanvi_reference/` — independent pretrained scANVI reference model, with registry and weight files

## Install

```bash
git clone https://github.com/YOUR_USERNAME/NSCLC-Regimen-MIL.git
cd NSCLC-Regimen-MIL
python -m pip install -r requirements.txt
python -m pip install -e .
```

Git LFS is required for model `.pt` files (`git lfs install && git lfs pull`).

## Single-cell MIL inference

Your H5AD must contain **one row per cell** and gene symbols compatible with the exact gene vocabulary. Provide a patient column for multi-patient files, or use one-patient-file mode. Input counts must be raw UMI integers; for `--scale log1p`, the matrix must already have been transformed with the **same normalization protocol** as the training data. Normalize before gene alignment to avoid renormalizing solely on shared genes.

```bash
python -m nsclc_regimen_mil.predict \
  --input path/to/new_patient_counts.h5ad \
  --patient-col patient --scale counts --target-sum YOUR_VERIFIED_TRAINING_TARGET \
  --model-dir models/mil_final159 \
  --reference-centroids models/mil_final159/reference_centroids.npz \
  --max-cells 1536 --repeats 3 \
  --output outputs/regimen_scores.csv
```

`--target-sum` is **mandatory** for counts and must be verified against the original training preprocessing. A prior external comparison used the **unverified estimate** `3565`; this does not establish that 3565 is the original training parameter.

Outputs include mean `raw_R_score` and SD, and optionally `Mean_R_similarity`, `Mean_NR_similarity`, `Exact_RminusNR`, SD and ranks (10 regimens per patient). `Exact_RminusNR` is the average cosine similarity to **actual regimen-matched R patients** minus the average cosine similarity to **actual regimen-matched NR patients**, not a response probability.

For an already-log1p data set (matching training normalization), specify `--scale log1p` instead of counts. For one patient/file, omit `--patient-col` and supply `--patient-id Patient01`.

For a quick benchmark on selected external patients:

```bash
python -m nsclc_regimen_mil.predict \
  --input /data/gj/project/SCVI/ExternalValidation/GSE179994/GSE179994_pre_Tcells_real_scRNA.h5ad \
  --patient-col patient --only-patients P20 P23 \
  --scale counts --target-sum 3565 --repeats 3 --max-cells 1536 \
  --reference-centroids models/mil_final159/reference_centroids.npz \
  --output outputs/demo_P20_P23.csv
```

The benchmark normalization setting above is illustrative only; compare it with the original study's actual preprocessing configuration before claiming numerical reproducibility.

## Optional scANVI annotations

```bash
python -m nsclc_regimen_mil.annotate_scanvi \
  --input new_patient_counts.h5ad --patient-col patient \
  --model-dir models/scanvi_reference \
  --epochs 30 --output outputs/new_patient_annotations.csv
```

Input must be raw integer UMI counts. Query adaptation updates the query-specific model in memory (not the reference folder); `scanvi_confident_label` uses predefined pmax 0.70 / top-two margin 0.15 thresholds. scANVI gene registration differs from the 28,742-gene MIL vocabulary. It is not required for MIL inference.

## Limitations

This is a **research demonstration**, not a medical device or personalized treatment recommendation. The original 159-patient cohort provides only outcomes under treatments actually received. The model lacks causal identification of outcomes under nine unobserved alternative regimens. New-patient scores may be sensitive to cell composition, gene coverage, sequencing platform, batch effects, normalization choices, and sampling seed. Cross-regimen score rankings are exploratory and must not be treated as calibrated clinical efficacy comparisons.

Only publish checkpoints, reference centroids, derived scANVI registration data, and code after checking original data usage licenses and privacy risk. No underlying single-cell patient matrices are included.

## References and licensing

Please add study and data accession citations and a suitable code license after permission review. A license is intentionally not invented by this automation.
