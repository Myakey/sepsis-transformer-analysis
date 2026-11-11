# Phase 0 plan: sepsis attention v4

Date: 2026-11-11. No code written yet. Waiting for your answers to section 5.

**Read:** `sepsis_softmax_pipeline_v3.ipynb` (all code, all aggregate outputs), `attention_method_8_points.pdf` (all 15 pages), the KA-Transformer paper (all), both EDA notebooks (aggregate outputs only). Reyna 2020 and Ramapuram 2025 were checked for specific facts only. v2 was not read; v3 supersedes it and I will open it when a v2 fact is needed.

## 1 · v3 numbers, re-verified

Every number in your state-of-play matches the v3 outputs: 74,704 stays, 1.146 % positive rows, 65.3 % early-onset, split sha1 `65441cff1a47`, soup AUROC 0.8073 / 0.8098 / 0.8050, AUPRC lift 6.2–6.4×, LightGBM 0.766, ICULOS +0.0004 (CI −0.0027 to +0.0034), Platt slope 0.98, band 0.3458, L4H3 −0.268, 0 of 32, 7 TOST-equivalent, I² ≈ 0.99. Three things the summary leaves out:

- The winning model is d_model 768, **4 layers × 8 heads**, lr 2.53e-5, batch 32, focal γ 1.0 / α 0.878, **aux_w 0.05**, 24.3 M parameters, trained on last-168 h windows only.
- The 12 largest trained effects all have confidence intervals that exclude zero, and all 12 replicate on SCREEN. The v3 result is "trained heads are not distinguishable from controls", not "no association". That is the S1 trigger, and it should be worded that way.
- The notebook shows only the top 12 heads per table, so the control maxima (untrained ≈ 0.25, shuffled ≈ 0.40) are confirmed for those rows only. The full tables are in `head_table_clr.csv` on Drive.

## 2 · Where the PDF and v3 disagree

| # | PDF | v3 | Consequence |
|---|---|---|---|
| 1 | Gate is **AUPRC well above base rate** | Gate is AUROC ≥ 0.85 | v3 already passes the PDF's gate (6×). Replacing 0.85 restores the pre-registered rule; it is not a rescue. |
| 2 | Shuffled-label control breaks the clinical signal | It is trained **with the auxiliary forecasting head** (aux loss 0.206 vs 0.19 in the real model) | The control still learns physiology, and it sets the top of the band (0.40). "Trained ≈ shuffled" then means "sepsis labels add nothing beyond physiology forecasting", a narrower claim than S1 as written. |
| 3 | "Carry the top-k forward" | Top-k **among heads that pass** → none | By the PDF text, top-6 by \|b1\| on DISCOVERY go forward. Two of those six are layer-1 heads, where effects on other hours cannot appear. |
| 4 | Untrained control: same architecture (your brief: same init) | Different seed (`SEED+777`) | v4 uses the trained model's own step-0 weights, which also allows a per-head trained-minus-init contrast. |
| 5 | Add ICULOS_t as a covariate when stays are truncated | Truncates to 168 h, omits it | Disappears with the full-length window (section 4). |
| 6 | Unit of inference is the subject when a subject has several stays | Stays pooled as independent | v4 clusters by subject. |
| 7 | Arms use an "identical schedule" **and** sigmoid gets 2× warmup | n/a | The PDF contradicts itself. I follow the 2× rule and add a softmax run with 2× warmup so warmup cannot explain an arm difference. |

## 3 · Risks in the design

1. **ICULOS is already in every model.** The positional encoding is a sinusoid of the absolute ICU hour; GRU-D's Δ for a never-measured feature equals hours since admission; and under a causal mask the number of visible keys is the hour. LightGBM gets it through the `*_hrs_since` features (its top feature is `FiO2_hrs_since`). The ±ICULOS ablation therefore tests only "one extra explicit ICULOS input", which is why it shows +0.0004. A Transformer with no time information cannot be built under a causal mask from hour 1.
2. **I² ≈ 0.99 is inflated.** Stage-1 standard errors treat every (t, j) pair as independent, but severity_j is repeated across up to 24 queries, so the errors are several times too small. Pooled estimates barely change; I² and τ do. The S5 trigger and the prediction intervals cannot rest on the current value.
3. **The KA paper solves a different task.** It is one prediction per patient (global average pooling, sepsis vs non-sepsis), ICU length of stay is an input and its top SHAP feature, and it reports a higher AUROC 12 h ahead (0.984) than 1 h ahead (0.962). The non-sepsis window is not defined. So the ladder has one step that changes the unit of evaluation from patients to hours; that step cannot be a paired bootstrap, and the waterfall should show it as a break. The kernel is also under-specified: what the distance is computed on, whether rows are normalised, what "kernel size 64" means, and 512 ÷ 6 heads.
4. **Sigmoid arithmetic at early hours.** σ(z − log T) gives 1/(1+T) at z = 0, so total mass is T/(T+1): 0.5 at hour 1 against 1.0 for softmax. The arms differ by construction exactly where 65 % of septic stays have their onset. The init test will check 1/(1+T), not 1/T. Ramapuram et al. stabilise sigmoid attention with LayerScale or a relative-position bias, which the matching rule forbids, so a failed gate is plausible and would be a limitation, not a finding.
5. **Heads are not aligned across seeds.** L4H3 in seed 42 is not L4H3 in seed 1042. With 5 seeds per arm, "pattern present" needs a seed-level rule (for example, heads beyond band per seed, then sign test across seeds), fixed in the pre-registration.
6. **VAL is overused.** It tuned our Transformer (30 trials), stopped training, built the soup and fitted calibration, while KA and the balanced variants get no tuning. Track A comparisons on VAL favour our model, and a VAL-chosen threshold scored on VAL is optimistic. I will cross-fit thresholds inside VAL and propose one frozen-model table on EXT at the very end.
7. **CONF is used twice.** `04` confirms on CONF, then `05` picks its candidate from those results and confirms on CONF again.
8. **PhysioNet cohort and label shift.** The Challenge excluded onset < 4 h after admission; MIMIC has 63 % of septic stays positive by hour 3. Stays positive on their first row have an onset known only to within 4–7 h. The Sepsis-3 operationalisation also differs from yours. The primary cross-cohort comparison should be on a matched stratum (onset ≥ 4 h, stay ≥ 8 h), with all-stays reported beside it.

## 4 · Calls I am making (logged; tell me if you object)

- `MAX_SEQ_LEN = 336`. The longest stay in MIMIC and in both PhysioNet sets is 336 h, so one window covers every stay, every row is predicted once with full history, and no overlapping-window machinery is needed. The v3 "last 168 h" view is reported as a bridge.
- The v3 estimator stays the primary decision rule. Cluster-robust stage-1 errors are reported beside it, and S5 uses the robust values.
- DISCOVERY and SCREEN stay IDs are loaded from v3's `correlational_summary.json`, not re-drawn.
- Seeds 42, 1042, 2042, 3042, 4042 for both arms.
- "Bitwise identical" is tested on CPU in fp32. On the T4 with fp16 autocast two runs of the same code already differ, so no bitwise claim is made there.
- My sandbox is CPU-only (torch 2.14.1, same version as Colab). The GPU and AMP paths are first exercised when you run `00_core_tests` in smoke mode.

## 5 · Questions for you

1. **Gate.** Adopt: for each arm's primary seed, at every horizon, AUROC CI lower bound above the LightGBM reference **and** AUPRC ≥ 3× base rate? v3 softmax passes (+0.041 over LightGBM, CI +0.028 to +0.054, mean over horizons).
2. **Shuffled-label control.** Keep the v3 protocol (aux head on) as the primary band member, and add an aux-off version and the budget-matched version as extra controls reported beside the band, never moving it?
3. **Carry-forward.** Top-6 by |b1| on DISCOVERY per the PDF text, re-ranked for each v4 model by that same rule, labelled "carried forward without passing the screen"?
4. **ICULOS for the lecturer.** Report "explicit ICULOS input on/off" for all models plus one time-blind LightGBM (no `hrs_since` for never-measured features), and state plainly that the Transformer always knows the hour?
5. **KA replication.** (a) Reproduce the paper first on PhysioNet, its own data, so the waterfall starts from a number we measured. (b) Non-sepsis window: full stay as the literal reading, with a length-matched control as the first rung. (c) Kernel: exp(−β‖q_i − k_j‖²) on per-head projections, learned β per head, rows normalised; unnormalised as a sensitivity run. (d) 8 heads × 64 = 512, reading "kernel size 64" as the per-head dimension. Agree with all four?
6. **CONF.** Split it by subject into CONF-A (`04`) and CONF-B (`05`) when the pre-registration is frozen?
7. **§10 is missing.** The brief ends at §9, in both the attachment and `Docs/FABLE_PROMPT_sepsis_v4.md`. `Resources/` holds 8 papers plus your PDF: Adebayo, Che, Jain & Wallace, KA, Ramapuram, Reyna, Singer, Wiegreffe & Pinter. Please send §10. Until then I cite only those.
8. **Colab path.** Where does this folder live on Drive, so the notebooks can import `Notebooks/v4/sepsis_core/` without manual steps?

## 6 · Phase 1, once you reply

Build `sepsis_core` and `00_core_tests`; run the eight required tests plus the 1/(1+T) init check on synthetic data with the exact schema; run the fresh-context reviewer agent; fix what it finds; hand over.

## Governance note

While extracting v3 outputs I printed one saved table that was row-level: 8 hourly rows for a single `stay_id` with model probabilities (no vitals, labs or demographics). It came from the calibration cell's demo `head(8)`. The same demo table is stored in the v2 and v3 notebook files (calibration cell), and the PhysioNet EDA notebook stores a `df.head()` output (cell 2). Clear those three outputs before the notebooks are shared or pushed. v4 notebooks will not display row-level frames.
