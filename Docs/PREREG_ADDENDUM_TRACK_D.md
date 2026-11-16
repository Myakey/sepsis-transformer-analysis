# Pre-registration addendum · Track D (notebook 05)

Addendum to `Docs/PREREGISTRATION.md` (frozen 2026-11-15, sha256 `e7f926d38bce…`), required by its section 10: "The choice, each candidate's primary explanation method and its hyperparameters are written to `Docs/PREREG_ADDENDUM_TRACK_D.md` and committed before any `05` training."

- Written 2026-11-16 by Claude on Miaki's instruction to build notebook 05. **In force once Miaki commits it**; notebook 05 refuses to run without it and records its sha256 in every output.
- Known when this was written: the full results of `03`, and the LOC stage of `04`. **Not known: CONF-A, CONF-B, EXT, and anything about the models trained in `05`** (none exists yet).
- Nothing in the frozen file is changed. Section 3 below lists every reading of a sentence that section 10 left open.

## 1 · The candidates are picked by rule, not by hand

Notebook 05 (Cell 2) reads `reports/03_summary.json` and `reports/04_summary.json` and applies the table of section 10 (`sepsis_core.fix.decision_table`). A row either fires or it does not.

| Row | Fires iff (read from the saved file) | Candidates | Status on 2026-11-16 |
|---|---|---|---|
| 1 | `03`: softmax rung-0 verdict is ABSENT | `vartok` (variable-level tokens), then `kernel` (kernel attention, "second") | **Fires** (E = −8, −7, −8, −8, −8) |
| 2 | `04`: a four-outcome cell is assigned and it is softmax Absent / sigmoid Present; **or** `03`: at the softmax primary seed at least one head has more than 5 % of its analysed weights below 1e-6 | `sigmoid` (the trained sigmoid arm as the explanation layer) | **Fires** on the floor clause (1 head of 32 at seed 42). The cell clause waits for CONF-A |
| 3 | `04`: softmax rung 1 supported, rungs 2 and 3 not (under the floor in force, deviation P1) | No new model: `softmax_dz` (row-centred logit) and `softmax_normw` (norm-weighted attention) on the standard softmax models | Waits for CONF-A |
| 4 | `03`: S5 trigger | Every explanation is reported with seed and bootstrap bands | **Fires**. The bands are reported in every case |

Two new architectures at most (section 10): `vartok` and `kernel`. Rows 2 and 3 add no architecture and need no training.

## 2 · Each candidate: model, primary explanation, screen DV, hyperparameters

| Candidate | Models (5 seeds: 42, 1042, 2042, 3042, 4042) | Primary explanation (hour level, for the t+6 logit) | DV in the section 5 screen | Controls for the band |
|---|---|---|---|---|
| `vartok` | `vartok_s{seed}`, **trained in 05** | **Additive contribution** of each hour to the logit (sum over that hour's measurements and over heads) | Each readout head's hour contribution, centred within the query row (`clr` slot of the screen) | own `init.pt`; `vartok_shuf_v3` (trained in 05, v3 protocol) |
| `kernel` | `kernel_s{seed}`; 42, 1042, 2042 exist from `02`, **3042 and 4042 trained in 05** | Raw attention, last layer, mean over heads | Row-centred logit (`clr`), as pre-registered for softmax-normalised rows | own `init.pt`; `kernel_shuf_v3` (from `02`) |
| `sigmoid` | `sigmoid_s{seed}` (from `02`) | Raw attention weight a, last layer, mean over heads | log a (`loga`), the sigmoid screening DV of section 5 | own `init.pt`; `sigmoid_shuf_v3` |
| `softmax_dz` | `softmax_s{seed}` (from `02`) | Row-centred logit, last layer, mean over heads | `clr` | own `init.pt`; `softmax_shuf_v3` |
| `softmax_normw` | `softmax_s{seed}` (from `02`) | Norm-weighted attention (Kobayashi et al. 2020), last layer | `normw` | own `init.pt`; `softmax_shuf_v3` |

**The `vartok` model** (`sepsis_core/fix.py`).

- Trunk: the matched softmax model, unchanged: same code and hyperparameters (`v3_best_hp.json`: d_model 768, 8 heads, 4 layers, FFN ×3, dropout 0.148, lr 2.53e-5, weight decay 4.99e-5, batch 32, focal loss, auxiliary next-hour head 0.05). For a given seed it starts from the same initial weights as `softmax_s{seed}` (checked and printed in Cell 4).
- Tokens (STraTS-style triplets, Tipirneni & Reddy 2022): one token per **measured** variable per hour = variable embedding + value embedding (1 → 16 → 256, tanh) + sinusoidal hour encoding; plus one token per hour meaning "this hour exists". Token width 256.
- Readout (the explanation layer): 4 attention heads from the trunk state of hour t over the tokens of hours ≤ t (softmax over tokens). The three logits **are** the weighted sum: logit = bias + Σ weight × (a number that depends on the token alone). Every prediction is therefore an exact sum of contributions of named measurements; the notebook checks this identity numerically.
- Training: identical to the primary arms of `02` (data, loss, schedule, up to 24 epochs, patience 8, soup of 5). No tuning. Readout width 256 and 4 heads are fixed here and not searched.
- Why not one flat sequence of all measurements, as in STraTS: up to 6,216 token slots per stay (168 hours × 37), so self-attention over them costs up to about 1,300× the hourly model per layer. The trunk stays hourly; the measurements enter at the readout.
- SOFA columns are neither inputs nor targets.

## 3 · Readings of section 10 that this addendum fixes

| # | Sentence in section 10 | Reading |
|---|---|---|
| a | "floor-effect fraction above 5 %" (row 2) | The quantity `03` already reports per model: the number of heads with more than 5 % of analysed weights below 1e-6. The row fires if that number is ≥ 1 at the softmax primary seed. **This reading was fixed after seeing the 03 value (1 head)**; the stricter reading "mean over heads above 5 %" would not fire. Consequence of the reading: `sigmoid` enters the scorecard, at no training cost. If Miaki prefers the stricter reading, say so before the VAL stage and the row is decided by the CONF-A cell alone |
| b | "sigmoid or entmax-1.5" | Sigmoid: five trained seeds and their controls exist. Entmax-1.5 would be a third new architecture |
| c | "explain with Δz or norm-weighted attention" (row 3) | Δz of a single prediction = the row-centred logit (the normaliser cancels); it and norm-weighted attention are two explanation-only candidates on the standard softmax models |
| d | Item 2, "PRESENT by the seed rule" | E = heads passing and confirmed in the trained model minus the larger of the untrained and shuffled-label counts, per seed; PRESENT iff E > 0 in 5 of 5 seeds (5.6). For models `03` already screened on VAL, its saved records are used |
| e | Item 3, which seed | Comprehensiveness is averaged over the five seeds per readout, then compared (random order on the same models; raw attention of `softmax_s{seed}`) with a subject-level paired bootstrap. Per-seed values are printed beside it |
| f | Item 3, readouts | 1,000 readouts (stay, hour ≥ 8 with a valid t+6 label), at most 3 per stay, drawn with a fixed seed; the same readouts for every model and method. Deletion = hours set to "not measured", ranked largest explanation value first |
| g | Item 4 | Mean pairwise Spearman correlation across the five seeds of the hour ranking, on the same readouts. Not recomputed on CONF-B (section 10 names items 1–3); the VAL value is carried |
| h | "Items 1–3 are recomputed once on CONF-B with everything frozen" | Frozen: candidates, explanations, models, controls, procedures, sampling seeds. On CONF-B the screen draws DISCOVERY and SCREEN stays by v3's procedure (800 each, split by subject) and builds its own band from the same controls; gate D1 uses the LightGBM model of `02` scored on CONF-B. One pass; a completion marker closes it |
| i | S6, "a candidate passes the scorecard on VAL and on CONF-B" | Read strictly: items 1–4 on VAL and items 1–3 on CONF-B. Reported beside it, with its own label: candidates that pass items 1 and 2 on both (the brief's wording, "passes the sanity checks the standard model fails, at non-inferior AUROC") |
| j | Methods compared on every candidate | Raw attention, clr, norm-weighted, rollout (softmax-normalised stacks only), Integrated Gradients, gradient × input; for `vartok` also its additive contribution. Primary seed, 300 of the readouts. Reported beside the scorecard; not a criterion |

## 4 · Order of work

1. This file is committed (with the approval of deviations P1 and P2).
2. `04`, stage CONF-A.
3. `05`, stage VAL: training, then the scorecard on VAL. The decision table is read at run time, so rows 2 and 3 use the CONF-A result.
4. `05`, stage CONF-B: one pass.
5. `04`, stage EXT: one pass.
