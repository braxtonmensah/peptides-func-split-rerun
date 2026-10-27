## Peptides-func: same model, two splits

Gate (PREREG.md): H1 if diff >= +0.03 and same sign in >= 4 of 5; H0 if diff < +0.01 or sign inconsistent; else INCONCLUSIVE.

| model | contrast | AP (shipped or twinned) | per-resample diff | mean diff | sd | t | verdict |
|---|---|---:|---|---:|---:|---:|---|
| M1 | A | 0.7460 | +0.1870, +0.1753, +0.1623, +0.1826, +0.1794 | +0.1773 | 0.0094 | 42.14 | **H1** |
| M2 | A | 0.3542 | +0.0338, +0.0280, +0.0274, +0.0275, +0.0396 | +0.0312 | 0.0054 | 12.98 | **H1** |
| M1 | C | 0.7966 | +0.1972, +0.1698, +0.2276, +0.2287, +0.2365 | +0.2119 | 0.0280 | 16.95 | **H1** |
| M2 | C | 0.3775 | +0.0545, +0.0728, +0.0549, +0.0607, +0.0713 | +0.0629 | 0.0088 | 16.00 | **H1** |

Controls: shipped twin_at_080 65.29% (1522 rows); corrected file and resamples corrected_file 0.0%, r0 0.0%, r1 0.0%, r2 0.0%, r3 0.0%, r4 0.0%.
Paper reproduction (M1 shipped vs 0.7460, tol 0.01): COMPARABLE (0.746043).
Shuffle: M1 AP 0.1656 vs base 0.1646, M2 AP 0.1802 vs base 0.1646.
Determinism: M1 True, M2 True.

Failures: none
