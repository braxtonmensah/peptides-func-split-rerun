# PREREG: does Peptides-func's near-twin contamination inflate a retrained model's score?

Written 2026-10-27 14:43 ET (US Eastern), before any model in this repository was trained or any
file under `results/` existed. The commit that adds this file must be older than every commit that
adds a result; the `prereg-check` job enforces that on every run.

## Why

The Silicyte leakage record measures a contamination rate: 65.29% of LRGB Peptides-func test rows
(1,522 of 2,331) have a training sequence at Levenshtein ratio 0.80 or above. A rate is not an
inflation. An earlier one-off run (W6, `inflation_lrgb.py`, ridge on 26 fixed descriptors) read
+0.0333 macro AP, but it ran once, on a laptop, with one weak model. This repository turns that into
a measurement anyone can rerun by forking and pressing Run workflow, and adds the published
fingerprint baseline.

## Gate, copied word for word from `vendor/inflation_lrgb.py` (pinned commit 8647e8c)

```
  H1  mean(original) - mean(corrected) >= +0.03 AP, with the same sign in at least 4 of 5
      resamples. That would be the first inflation figure on this benchmark.
  H0  below +0.01, or sign inconsistent. Then LRGB's contamination costs nothing measurable for
      this model, which is the same answer W1 gave on PepBenchmark, and two nulls in a row would
      make "contaminated but it costs less than you think" the honest headline of this programme.
  Between is inconclusive and gets reported as inconclusive.
```

(The phrase "the first inflation figure" is copied as written in October and is not a claim this
repository makes.)

Machine-readable form, parsed by the code that computes every verdict. `prereg-check` fails if
these values differ from the ones parsed out of the vendored docstring.

```gate
{"h1_min_diff": 0.03, "h1_min_same_sign": 4, "n_resamples": 5, "h0_below": 0.01}
```

Verdict rule, applied identically to every contrast: with `d` the 5 per-resample differences,
`same` = number of `d` whose sign equals the sign of `mean(d)`:
H1 if `mean(d) >= 0.03` and `same >= 4`; otherwise H0 if `mean(d) < 0.01` or `same < 4`;
otherwise INCONCLUSIVE. This is the exact expression in `inflation_lrgb.py`.

## Data, fixed now

- LRGB's own files, from the URLs in LRGB's loader: `peptide_multi_class_dataset.csv.gz` (MD5
  `701eb743e899f4d793f0e13c8fa5a1b4`) and `splits_random_stratified_peptide.pickle` (MD5
  `5a0114bdadc80b94fc7ae974f13ef061`). These MD5s are the ones recorded in the Silicyte record
  (header of the corrected split file and silicyte.com/data/lrgb-corrected-v1/) and in LRGB's
  loader. A mismatch fails the job.
- Cross-check: the scikit-fingerprints Hugging Face copy (`scikit-fingerprints/LRGB_Peptides-func`,
  read through `skfp.datasets.lrgb`, no account) must carry the same SMILES in the same order and
  the same split indices. This is the loader arXiv 2501.17901 used.
- Shipped split sizes must be 10,873 / 2,331 / 2,331. The corrected split file
  (`lrgb_peptides_func_singlelinkage080.json`, silicyte-site commit a147b98, MD5 as served
  `b5751ec51e80fa7fb5aba12016de81fb`) must have the same sizes.
- Five corrected resamples are rebuilt in CI with seeds 0, 1, 2, 3, 4 using `build_corrected()` from
  the vendored `inflation_lrgb.py` over clusters from the vendored `build_lrgb_split.py`
  (transitive single linkage, union-find, Levenshtein ratio 0.80). The published corrected file is
  scored too and reported, but it is not one of the five gated resamples.

## Models, fixed now, CPU only

- **M1, primary.** The published fingerprint baseline from arXiv 2501.17901 (Adamczyk, Ludynia,
  Czech): count ECFP from scikit-fingerprints (`ECFPFingerprint(count=True)`, library defaults,
  radius 2) into one `LGBMClassifier(n_estimators=500, is_unbalance=True, importance_type="gain",
  random_state=0, verbose=-1)` per label (`MultiOutputClassifier`), as in the authors' code
  (github.com/scikit-fingerprints/peptides_molecular_fingerprints_classification, commit a76d135).
  Trained on train + val, scored on test, exactly as that code does. Versions pinned to the
  authors' `uv.lock`: scikit-fingerprints 1.18.0, lightgbm 4.6.0, scikit-learn 1.8.0, rdkit
  2025.3.6, numpy 2.4.1, Python 3.12. The only addition is `deterministic=True` and a fixed thread
  count, which change no modelling choice.
- **M2, continuity.** The 26-descriptor ridge averaged over a fixed penalty grid, imported unchanged
  from the vendored `inflation_pepbench.py` / `inflation_lrgb.py`. Trained on train only, as W6 did.

Metric: average precision per label, macro-averaged over the 10 labels (`sklearn`
`average_precision_score`), which is what LRGB reports.

## Contrasts

**A (primary for M1; continuity for M2).** Shipped split vs corrected split. `d_i` = macro AP on the
shipped split minus macro AP on corrected resample `i`, i = 0..4. The shipped split is fixed and both
models are deterministic, so the shipped score is one number and the spread is split-assignment
noise. Reported: per-resample AP, mean, sd, mean difference, one-sample t on `d` (paired against the
fixed shipped score), verdict.

Known confound of A, stated before the run: the corrected test set is a different set of rows, with
different label rates and possibly different difficulty. A cannot separate "contamination helped"
from "the new test rows are harder".

**C (same training, different test rows).** M1 and M2 trained on the shipped training set, scored
on the shipped test set split into twinned rows (a training-set sequence at ratio 0.80 or above) and
untwinned rows. Twinned is measured against the rows each model actually fits on: train (1,522
twinned of 2,331, the record's number) for M2, train + val for M1 (its count is reported). Because
the untwinned group is smaller, each of the 5 resamples draws, with seed i, a random subset of the
twinned rows of the same size as the untwinned group; `d_i` = macro AP(twinned subset i) minus macro
AP(untwinned). Labels with no positive in either group are dropped from both, and the count is
reported. The same gate decides it.

Known confound of C, stated before the run: twinned and untwinned rows are not the same kind of
row. Peptides from large families may be easier or harder for reasons other than having a relative
in training, and AP depends on each group's positive rate. C removes the different-training-set
confound of A; it does not remove row difficulty. Per-group positive rates are reported next to it.

## Controls (any failure fails the run, except where marked)

1. Leakage control in the same job: shipped split twin_at_080 = 65.29% and 1,522 twinned test rows;
   corrected file and each of the 5 resamples = 0.00%.
2. Label-shuffle (must-fail): each model trained on row-permuted training labels (seed 0) on the
   shipped split must score a macro AP within 0.02 of the mean test positive rate across the 10
   labels. Per-label deviations are reported; any above 0.04 is flagged in the summary.
3. Paper reproduction (flag, not fail): M1 on the shipped split must reproduce the 74.60 AP that
   arXiv 2501.17901 reports for count ECFP (Table 1) within **0.010** AP (0.7360 to 0.7560). Outside
   that band every M1 number is labelled NOT COMPARABLE to the paper; the contrast is still reported.
4. Determinism: M1 and M2 on resample 0 run twice in one job must agree to 6 decimals.

## Verdict that goes on the badge and the page

The badge and the headline carry **M1, contrast A**. M2 contrast A and both contrast C verdicts are
reported beside it with equal prominence in `results/summary.json` and the README. Whichever way it
falls (H1, H0 or INCONCLUSIVE) is published.

## What this will not establish

It measures two named models on one benchmark. It does not say what any graph network or
transformer on the LRGB leaderboard loses, and the number must be quoted with the model named.
