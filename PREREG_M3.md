# PREREG_M3: arm M3, a tuned graph network, on the same splits

Written 2026-10-27 15:26 ET (US Eastern), before any M3 code ran and before any M3 number existed.
This file is an addendum to [PREREG.md](PREREG.md). It changes nothing about M1 or M2: their code,
their numbers and their verdicts stay as published in run 40564969278. The commit that adds this
file must be older than every commit that adds a file under `results/m3/`; `src/m3_prereg_check.py`
enforces that, and also that the last commit touching this file is older than every M3 result.

## Why

The M1 and M2 results are for a fingerprint model and a descriptor ridge. The README says plainly
that they do not say what a graph network from the LRGB leaderboard loses. M3 tests that directly
with one leaderboard-class graph network.

## M3, fixed now

The tuned GCN of Tönshoff, Ritzert, Rosenbluth, Grohe, "Where Did the Gap Go? Reassessing the
Long-Range Graph Benchmark" (TMLR 2024, arXiv 2309.00367), config file
`configs/LRGB-tuned/peptides-func-GCN.yaml` in github.com/toenshoff/LRGB at commit
`c3500cb702dcc5bdbe02783b94f24eebb7fb52c9` (2024-04-03). The paper reports 0.6860 ± 0.0050 test AP
for this model on the shipped split (mean ± sd over 4 seeds).

Their code runs inside GraphGPS/GraphGym. M3 is a reimplementation of that config in plain
PyTorch Geometric, so that it can be trained on arbitrary split indices in CI. Everything below is
read from the config and from the layer, encoder, head, loss, scheduler and training-loop files at the
same commit:

- Graphs: `ogb.utils.smiles2graph` on LRGB's SMILES, as their dataset loader does (9 atom features).
- Node encoder `Atom+RWSE`: OGB atom embeddings (sum of 9 embeddings, xavier init) to 235 - 28 = 207
  dims, concatenated with RWSE: random-walk landing probabilities for k = 1..20 (`P = D^-1 A`, the
  diagonal of `P^k`, as `get_rw_landing_probs` computes them), `BatchNorm1d(20)` raw norm, then
  `Linear(20, 28)`. Total width 235.
- 6 message-passing layers, each `GCNConv(235, 235)` (PyG defaults: self-loops, symmetric norm),
  then GELU, then dropout 0.1, then a residual add, as in `graphgps/layer/gcn_conv_layer.py`.
- The config also builds a `Bond` edge encoder. `GCNConv` never reads edge features, so it cannot
  change any output; M3 omits it. This is stated as a deviation.
- Head `mlp_graph`: mean pooling, then (dropout 0.1, Linear 235->235, GELU) twice, then dropout 0.1
  and Linear 235->10.
- Loss: `BCEWithLogitsLoss` over the 10 labels. AdamW, lr 0.001, weight decay 0, gradient-norm clip
  1.0, batch size 200, training rows shuffled each epoch.
- Schedule: linear warmup for 5 epochs then cosine decay to 0 at `max_epoch` (the Hugging Face
  `get_cosine_schedule_with_warmup` they call), stepped once per epoch.
- Every epoch: macro AP on validation and test. The reported test score is the one at the epoch with
  the highest validation macro AP (first such epoch), as their `custom_train` selects with
  `metric_best: ap`.
- Fit on **train only**; val is used for epoch selection, test is scored once per epoch and never
  used for any choice.
- CPU only, `torch.use_deterministic_algorithms(True)`, 4 threads, seeded with `torch.manual_seed`,
  `numpy` and `random` set to the job's seed.

Versions pinned in `requirements_m3.txt` (torch 2.5.1 CPU wheel, torch_geometric 2.6.1, ogb 1.3.6,
rdkit 2025.3.6, Python 3.12). These are not the versions in their environment; that is a deviation.

## Compute budget: fixed by a timing-only probe, before any metric

GitHub-hosted jobs stop at 6 hours. The config trains for 250 epochs. Whether that fits on a 4-vCPU
runner is unknown when this is written. Rule, fixed now:

1. One probe job (`m3.yml`, input `mode: probe`) builds the graphs, trains 3 epochs on the shipped
   training rows and runs 3 forward passes over val + test. It computes **no metric of any kind**:
   the probe code path never calls a scoring function and its output JSON carries only seconds. It
   prints setup seconds, seconds per training epoch and seconds per evaluation pass.
2. From the probe: projected job time = setup + E x (train epoch + eval pass) x 1.15 (margin for
   slower runners). If E = 250 projects to 330 minutes or less, M3 uses the full 250 epochs.
   Otherwise E is the largest multiple of 10 that projects to 330 minutes or less, and the cosine
   schedule runs over E epochs (warmup stays 5). A reduced budget is a deviation from the published
   config and every M3 number will be labelled with E.
3. E is written into the `m3budget` block below, with the probe's run URL and its seconds, in a
   commit pushed **before** the run that produces M3 numbers. `src/run_m3.py` reads E from this
   block; it refuses to train if E is not set.

```m3budget
{"max_epoch": null, "probe_run": null}
```

## Jobs, fixed now

| split | seeds | mode |
|---|---|---|
| shipped | 0, 1, 2 | normal |
| r0, r1, r2, r3, r4 (the 5 corrected resamples, rebuilt by the same `prepare.py`) | 0 | normal |
| corrected_file (published corrected split, not gated) | 0 | normal |
| shipped | 0 | shuffle (training labels row-permuted with numpy seed 0) |
| r0 | 0 | determinism (a second, separate job; compared with the r0 normal job) |

Eleven jobs, all with the same E.

## Contrasts and the gate

Metric: average precision per label, macro-averaged over the 10 labels (sklearn
`average_precision_score`), the same function M1 and M2 use, at the selected epoch.

**Contrast A (primary for M3).** `d_i` = mean test macro AP over the 3 shipped-split seeds minus the
test macro AP on corrected resample `i` (seed 0), i = 0..4. Reported: per-resample AP, mean, sd,
mean difference, one-sample t on `d`, verdict. The 3-seed sd on the shipped split is reported as the
seed-noise reference.

**Contrast C (secondary).** Shipped split, seed 0, the same training: test rows with a training
sequence at Levenshtein ratio 0.80 or above (twins counted against **train**, the rows M3 fits on:
1,522 twinned, 809 untwinned) versus test rows without one, twinned rows subsampled to the
untwinned count with seeds 0..4, labels with no positive in either group dropped. Same code path
(`contrast_c` in `src/run_arm.py`) as M1 and M2.

**Gate**: the same `gate` block in PREREG.md, read by the same `verdict()` function, applied to both
M3 contrasts. H1 if mean(d) >= +0.03 and the same sign in at least 4 of 5; H0 if mean(d) < +0.01
or the sign is inconsistent; otherwise INCONCLUSIVE.

**What counts as a null.** H0 on contrast A: for this GCN, the shipped split's contamination buys
less than 0.01 macro AP over splits with no test sequence twinned in training (or the direction is
not consistent). That would be published as the headline for M3 with the same prominence as an H1,
next to M1's +0.1773. INCONCLUSIVE is published as INCONCLUSIVE.

## Expected direction and band (written now, so the miss can be scored later)

Direction: positive on both contrasts. Point guess for contrast A: +0.12 macro AP, band +0.06 to
+0.18. Reasoning: M1 read +0.1773 and M2 +0.0312; a GCN on atom graphs should exploit near-twin
memorisation less than a count fingerprint with 500-tree boosting, more than a 26-descriptor ridge.
Contrast C point guess +0.15, band +0.08 to +0.25. If the result lands outside a band, that is a
calibration miss and the README will say so; it does not change the verdict.

## Controls

1. Data and leakage controls: inherited unchanged from `prepare.py` (MD5s, the scikit-fingerprints
   cross-check, sizes, shipped 65.29% / 1,522 twinned, corrected file and resamples 0.00%). Any
   failure fails the run.
2. Label-shuffle (must-fail): the shuffle job's test macro AP must be within 0.02 of the mean test
   positive rate over the 10 labels. Failure fails the run.
3. Paper reproduction (flag, not fail): the 3-seed shipped mean is compared with 0.6860. COMPARABLE if
   within 0.015 (0.6710 to 0.7010); otherwise every M3 number is labelled NOT COMPARABLE to the
   paper. With a reduced E, the label is reported together with E.
4. Determinism (flag, not fail): the r0 determinism job must equal the r0 normal job to 6 decimals.
   Two jobs can land on different CPU models and float reductions can differ, so a mismatch is
   reported with its size and flagged; it is not a failure. A mismatch larger than 0.01 AP is
   labelled in the README next to the M3 result.
5. Every one of the 11 jobs must finish all E epochs and write its output, else the aggregate fails.

## What this will not establish

One graph network, one config, on one benchmark, with at most 3 seeds on the shipped split and one
seed per corrected split. It does not speak for transformers, GatedGCN, GINE or GPS, or for the
leaderboard as a whole. Contrast A still compares different test rows and contrast C still does not
remove row difficulty (the same confounds PREREG.md states). A reduced epoch budget, if needed,
means M3 is the tuned GCN architecture under a shorter schedule, not the published run.
