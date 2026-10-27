# peptides-func-split-rerun

[![rerun](https://github.com/braxtonmensah/peptides-func-split-rerun/actions/workflows/rerun.yml/badge.svg)](https://github.com/braxtonmensah/peptides-func-split-rerun/actions/workflows/rerun.yml)
[![verdict](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbraxtonmensah%2Fpeptides-func-split-rerun%2Fresults%2Fbadge.json)](https://github.com/braxtonmensah/peptides-func-split-rerun/blob/results/results/summary.json)

**Same model, two splits.** 65.29% of the LRGB Peptides-func test set (1,522 of 2,331 rows) has a
training sequence at Levenshtein ratio 0.80 or above. This repository measures whether that
contamination inflates a retrained model's score, against a gate written in [PREREG.md](PREREG.md)
and committed before any model ran. Everything runs in GitHub Actions on free CPU runners.

**Rerun it in one line:** fork this repo, open the Actions tab, choose **rerun**, press **Run workflow**.
(No secrets, no accounts. GitHub asks you to enable Actions on a fork once.)

## What it does

| step | what | gate |
|---|---|---|
| prereg-check | PREREG.md was committed before any file under `results/`; its thresholds (+0.03, 4 of 5, +0.01) equal the ones parsed from the vendored `inflation_lrgb.py` | fails the run |
| prepare | downloads LRGB's own data and split files and checks their MD5s; checks the scikit-fingerprints Hugging Face copy has the same SMILES, labels and split; checks sizes 10,873 / 2,331 / 2,331; rebuilds 5 corrected splits (seeds 0 to 4) with the vendored single-linkage builder; measures the twin rate of every split in the same job (shipped must read 65.29%, corrected 0.00%) | fails the run |
| arms | a matrix of model x split, 18 jobs | |
| aggregate | contrasts, paired t, verdicts computed from the prereg thresholds, schema check, badge | controls fail the run |

**Models.** M1 is the published fingerprint baseline of arXiv 2501.17901 (Adamczyk, Ludynia, Czech):
count ECFP from scikit-fingerprints and one LightGBM per label (500 trees, `is_unbalance`), versions
pinned to the authors' lock file, trained on train + val as their code does. M2 is the 26-descriptor
ridge from the earlier W6 run, vendored unchanged, trained on train.

**Contrast A.** Macro AP (10 labels) on LRGB's shipped split minus macro AP on each of 5 corrected
splits. **Contrast C.** The same model trained once on the shipped training set, scored on the
shipped test rows that have a training twin versus those that do not (twinned rows subsampled to the
untwinned count, 5 seeds).

**Controls.** Label-shuffled training must land within 0.02 of the base rate (must-fail). Resample 0
run twice must agree to 6 decimals. M1 on the shipped split is compared with the 74.60 AP the paper
reports (tolerance 0.010); outside it, M1 numbers are labelled NOT COMPARABLE rather than passing.

## Result

The latest verdict is on the badge above and in
[`results/summary.json`](https://github.com/braxtonmensah/peptides-func-split-rerun/blob/results/results/summary.json)
on the `results` branch (job summary table in each run). Tagged releases attach the JSON.


## Limits

- Two named models on one benchmark. It does not say what any graph network or transformer on the
  LRGB leaderboard loses; quote any number with the model named.
- **Contrast A** compares different test sets: the corrected test rows differ in label rates and
  possibly difficulty, so A cannot separate "contamination helped" from "these rows are harder".
- **Contrast C** keeps training identical, which removes that confound for its comparison, but
  twinned and untwinned rows are not the same kind of row: peptides from large families may be easier
  or harder for reasons other than having a relative in training, and AP moves with each group's
  positive rate (reported beside it). C does not remove row difficulty.
- Of the 180 test rows that share a primary sequence with training, only 30 share a SMILES; a
  sequence-level twin is not always the same molecule.
- The 0.80 Levenshtein threshold is a choice; the corrected splits are defined at that threshold.
- LightGBM runs with `deterministic=True`, `force_col_wise=True` and 4 threads so reruns are
  bit-identical; these are compute options, not modelling choices, but they are not in the paper's code.
- LRGB data is CC BY-NC 4.0 and is downloaded at run time from LRGB's own URLs; it is not
  redistributed here. Only aggregate metrics are published.

## Sources

- Dwivedi et al., Long Range Graph Benchmark, NeurIPS 2022 Datasets and Benchmarks (arXiv 2206.08164).
- Tönshoff, Ritzert, Rosenbluth, Grohe, "Where Did the Gap Go? Reassessing the Long-Range Graph
  Benchmark", TMLR 2024 (arXiv 2309.00367). A critique of LRGB on hyperparameters, not on the split.
- Adamczyk, Ludynia, Czech, "Molecular fingerprints are strong models for peptide function
  prediction" (arXiv 2501.17901), code at scikit-fingerprints/peptides_molecular_fingerprints_classification.
- Contamination record and corrected split: [silicyte.com/benchmark-leakage](https://silicyte.com/benchmark-leakage),
  [silicyte.com/data/lrgb-corrected-v1](https://silicyte.com/data/lrgb-corrected-v1/), scripts in
  [silicyte-design-loops/thirdparty](https://github.com/braxtonmensah/silicyte-design-loops/tree/main/thirdparty).
  Checkable data-quality and leakage-aware splitting tools exist elsewhere (for example DataSAIL and
  GraphPart); this repository is a measurement, not a new splitting method.

## License

MIT, see [LICENSE](LICENSE). Built by Silicyte (silicyte.com) with AI assistance.
