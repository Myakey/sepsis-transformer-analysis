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

## Approved after the freeze (in force)

Found while building `04` on 2026-11-15, after the freeze and before any data of the causal stage was seen. Approved by Miaki on 2026-11-15, after the LOC stage of `04` and **before CONF-A was loaded**. Both were logged as proposals before the LOC run; the LOC output (descriptive only) showed the P1 problem as predicted: IIA(M0) 0.19 against a frozen floor of 0.43 (softmax, primary seed), 0.11 against 0.22 (sigmoid).

| # | Date | Rule affected | Change | Reason | Status |
|---|---|---|---|---|---|
| P1 | 2026-11-15 | `PREREGISTRATION.md` 7.7 (floor) and 7.9 (rung 1: "IIA(M0) beats the floor") | For a model that predicts a **constant** vector (M0 always; M2 and, mostly, M1 under do(Priority)), take the floor from a random key alignment: the same match rule with the direct effect d read at a random key of S instead of at k. The floor for models that predict structure stays the permuted-log-P floor. | The frozen floor permutes log P within S. For a constant-prediction model the match is "no rank structure, and d has the predicted sign"; permuting log P leaves both parts unchanged when there is no structure, so the model's IIA equals its floor by construction and **rung 1 can never be met as written**. This is an error in the draft I wrote, found before any data was seen. | **Approved: Miaki, 2026-11-15**, before CONF-A. The CONF-A run of `04` sets `IIA_FLOOR_CONSTANT = "random_key"` in Cell 2. Cell 14 of `04` computes the frozen floor beside it and saves both (`M0_beats_floor` under the floor in force; the frozen floor in `iia_alternative_floor` and `M0_beats_floor_under_the_other_floor`), so both are reported. The LOC choices file is unchanged: the floor is not a LOC choice, and the LOC IIA values are descriptive. |
| P2 | 2026-11-15 | `PREREGISTRATION.md` 7.8 (DAS "held-out IIA") | Define a DAS match: the patched t+1 logit moves in the direction of the source run and by at least half as much; pairs whose source effect is below 5 % of the SD of the base logit are dropped. Site: residual stream entering the layer chosen on LOC (largest whole-component restoration among layers ≥ 2), hour k; target of training: the three output logits of the source run. | 7.8 fixes the controls, the margin and the subspace sizes but not what counts as a match for a learned subspace. DAS runs only if no hand-specified model beats the floor. | **Approved: Miaki, 2026-11-15**, before CONF-A. DAS still runs only if no hand-specified model beats the floor; its output is no longer provisional. No code change. |
