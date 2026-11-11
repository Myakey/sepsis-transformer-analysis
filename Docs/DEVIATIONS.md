# Deviations log

Every change to a pre-registered rule is recorded here with its reason and date. Nothing is changed silently.
Rules in force: `Resources/attention_method_8_points.pdf` (treated as pre-registered) and, once frozen, `Docs/PREREGISTRATION.md`.

| # | Date | Rule affected | Change | Reason | Approved |
|---|---|---|---|---|---|
| D1 | 2026-11-11 | Performance gate before any interpretability run | The v3 notebook's gate "VAL AUROC ≥ 0.85" is retired. Gate in force: for each arm's primary seed, at **every** horizon, (i) the lower 95 % bound of AUROC(model) − AUROC(LightGBM reference) is above 0 on the same VAL rows (stay-level paired bootstrap) and (ii) AUPRC ≥ 3× the base rate. Implemented in `sepsis_core.metrics.gate`. | **This restores the PDF's rule; it does not replace a pre-registered one.** PDF P4 states the gate as "held-out AUPRC must be well above the base rate". The 0.85 AUROC figure was a v3 engineering target, never in the PDF, and LightGBM on the same rows reaches only 0.766, so 0.85 says nothing about whether the model is good enough to analyse. The new wording makes "well above" numeric (3×) and adds a reference-model condition. v3 softmax: AUPRC lift 6.2–6.4× at all horizons; AUROC +0.041 over LightGBM (CI +0.028 to +0.054, mean over horizons; the per-horizon check runs in `02`). | Miaki, 2026-11-11 |

## Not deviations (recorded so nobody mistakes them for one)

- The control band, the DV, the SESOI (0.05), the band quantile (P95) and `CORR_TOP_K = 6` are unchanged from v3.
- Extra controls (aux-off shuffled-label, budget-matched shuffled-label) are reported **beside** the band and never enter or move it.
- Kernel attention inside the trunk is an exploratory third variant, outside the pre-registered softmax-vs-sigmoid contrast.
