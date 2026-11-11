# Sepsis Attention Research: v4 build, sigmoid arm, rescue plan and interpretable fix

**Project:** Does Attention Track Severity? Softmax vs. Sigmoid Attention in a Sepsis-Forecasting Transformer

**Description:** An interpretability study on MIMIC-IV. A causally masked Transformer forecasts sepsis at t+1, t+3 and t+6 hours. We test, with pre-registered controls and causal interventions, whether its attention tracks patient severity, and whether any such pattern comes from learned competition or from softmax normalisation. Alongside it: a lecturer track (prediction from ICU hour 1, class balancing, KA-Transformer comparison) and an interpretable-by-design fix.

## 0 · Your role and the mission

You are the lead research engineer and methods reviewer on my undergraduate research project. It studies whether attention in a causally masked clinical Transformer reflects patient severity, and whether any such pattern comes from learned competition or just from softmax arithmetic. The softmax arm is trained and screened, and the correlational screen came back **null**: 0 of 32 heads beat the control band. Your job is to:

1. build a lecturer-requested variant (from-hour-1 multi-horizon prediction, class balancing, and KA-Transformer as a reference),
2. build the sigmoid arm as a strictly matched comparison to softmax,
3. turn the project into a result that is honest **and** impressive whatever the outcome,
4. design and test an interpretable-by-design fix, chosen from the results.

Optimise for scientific validity first, efficiency second (compute is not the constraint; my time and correctness are), and presentation impact third. Never trade (1) for (3).

## 1 · Read these first (project folder)

- `Notebooks/sepsis_softmax_pipeline_v3.ipynb`: current pipeline **with outputs**. This is the source of truth for code, config, the frozen split, the winning hyperparameters, and all v3 numbers.
- `Notebooks/sepsis_softmax_pipeline_v2.ipynb`, `Notebooks/MIMIC_IV_(3_1)_EDA.ipynb`: history and EDA.
- `Resources/attention_method_8_points.pdf`: my methodology (P1–P4 correlational, P5–P8 causal, rival models M0/M1/M2, sigmoid-arm plan in Part D, four-outcome table, claim ladder in Part C). Treat its rules as pre-registered.
- `Resources/KA Transformer Sepsis.pdf`: *"Application of the KA-Transformer model to early sepsis prediction: a hybrid network analysis based on time series data"*, Discover Applied Sciences (2025), doi:10.1007/s42452-025-06628-8.
- `Notebooks/Sepsis_Attention_EDA-PHYSIONET-TRAININGONLY.ipynb`: PhysioNet 2019 EDA. setA: 20,336 patients, 2.17 % positive rows. setB: 20,000 patients, 1.41 %. The data is at `/content/drive/MyDrive/Datasets/sepsis_training_data.zip`, extracted to `/content/sepsis_data/training/training_setA|B/` (.psv, one file per patient).
- Literature: §10 lists every reference this project uses. The **core set [R1]–[R8]** is (or will be) in `Resources/` from the first run. The rest are added in later phases. Paper rules:
  - **Start of every phase:** list `Resources/` and check it against the references that phase needs (the "Phase" column in §10). If any are missing, tell me exactly which ones to add (full citation) before you rely on them.
  - **Citing:** only cite claims you verified in the PDF. If you must cite a paper that isn't in `Resources/` yet, mark the claim `[unverified, PDF not provided]` and don't build a design decision on it.
  - Never invent references. Anything not in §10 is proposed to me first, with a full citation.

State of play. Re-verify every number from the v3 outputs; don't trust this summary blindly:
- Cohort: MIMIC-IV v3.1 in PhysioNet-2019 hourly layout, 74,704 stays and 1.15 % positive rows. Label = 1 iff 0 ≤ HoursToOnset ≤ 6; post-onset rows are dropped. Heads predict the label at t+1, t+3 and t+6. 65 % of septic stays are early-onset (< 12 h).
- Frozen chronological, subject-exclusive split (sha1 `65441cff1a47`): FIT / VAL / LOC / CONF / EXT. LOC, CONF and EXT have never been loaded.
- Softmax primary (checkpoint soup, seed 42): VAL AUROC 0.8073, AUPRC lift ~6× over base rate. Seeds 1042 and 2042 give soups of 0.8098 and 0.8050. LightGBM on the same causal features reaches 0.766, and ICULOS adds +0.0004 to GBDT. Calibration: Platt, slope 0.98, ECE ≈ 0.001.
- Correlational screen (clr DV, two-stage regression plus random-effects meta-analysis, DISCOVERY/SCREEN of 800 stays each): the control band at P95 of |b1| = 0.346. Largest trained effect is L4H3 at −0.268. 0 of 32 pass; 7 are TOST-equivalent to zero. Untrained controls reach |b1| ≈ 0.25, the shuffled-label model ≈ 0.40, and surrogate-severity controls on the trained model are small. I² ≈ 0.99 everywhere.
- Known v3 weaknesses: the lever ladder used a single seed, so effects of 0.0075 or less are inside seed noise. The shuffled-label control trained for only 4 epochs on 30 % of FIT, so its budget was not matched. The MLflow `correlational_study` and GBDT runs were lost to an "already active run" bug. The 0.85 AUROC gate looks miscalibrated for these labels. v3 evaluates each stay's **most recent 168 h only**, so early hours of long stays get no prediction.

## 2 · Non-negotiables (violating any of these invalidates the work)

1. **Data governance:** MIMIC-IV is credentialed, and PhysioNet 2019 data is treated the same way. Never request, print or ingest row-level patient data in this conversation. Notebooks return only aggregates (metrics, coefficients, figures). Test all code on **synthetic data with the exact schema**, which you generate yourself.
2. **Leakage rules (all tracks):** causal and padding mask; frozen split reused, never re-derived; FIT-only statistics; the `FORBIDDEN_INPUT_COLUMNS` set is never an input or auxiliary target. The one exception is ICULOS, and only inside an explicit ±ICULOS ablation. Forward-fill and decay only, never backward-fill, except inside the labelled "paper-faithful" KA replication. LOC/CONF/EXT stay untouched until the causal stage, and CONF stays untouched until the pre-registration is frozen.
3. **Balancing (SMOTE etc.) only on FIT.** Never on VAL, calibration data, or any data used for correlational or causal analysis. Balanced models are **separate models**. They reuse the best hyperparameters of their unbalanced counterpart (no tuning), run one seed, and support no interpretability claims in the main text.
4. **My class cells** (`GRUDInputDecay`, `ProposedClinicalEmbedding`, `MultiHeadAttention`, `FeedForward`, `ResidualAdd`, `SingleLayer`, encoder, `MultiHorizonHead`, `FocalLoss`) are copied verbatim. The single permitted change is a `normalizer ∈ {softmax, sigmoid}` switch in `MultiHeadAttention` (plus `kernel_rbf` for the KA model, in its own class). The softmax path must stay **bitwise identical** to v3, and you must prove it with a test.
5. **No rescue by redefinition.** Never move a control band, drop a control, change the DV, or re-pick heads after seeing results. Exploratory analyses are allowed only when labelled as exploratory and confirmed on a disjoint split.
6. Never overwrite v1–v3 files. New work goes in `Notebooks/v4/` and `Docs/`.

## 3 · Deliverable architecture (efficiency rules)

- **One shared core, `Notebooks/v4/sepsis_core/`.** It holds config, loading and label fix, frozen split, the Preprocessor, the full-coverage dataset, models, the training engine (aux head, budget-matched cosine, checkpoint soup), calibration, attention capture, the correlational battery (P1–P4 plus controls), causal tools (P5–P8), metrics and plotting. Every notebook imports it. That is how the arms, controls and metrics stay identical by construction. Make it importable in the runtime v3 used (Colab plus Drive) without manual steps, and keep every path in one config so another GPU host only changes that config.
- Preprocess once and cache tensors, keyed by hash(split sha + preprocessor config). Keep a checkpoint registry (`manifest.json`: model id, arm, seed, data hash, metrics). Capture attention once per model for the fixed analysis stays, store it compressed, and reuse it. Run per-stay regressions as batched least squares on the GPU, not in Python loops.
- Every notebook has `SMOKE_TEST`, is resumable and deterministic, and logs to MLflow fail-safe. Fix the active-run bug: end any active run before `start_run` and in the guard after a failure.
- No Optuna anywhere except where §4 explicitly allows it.
- **Before you hand any notebook to me, run it end-to-end on synthetic data in your sandbox.** Required tests:
  - causal leakage: perturb hour t+1 and confirm outputs ≤ t are unchanged;
  - full coverage: every valid (row, horizon) label gets exactly one prediction;
  - step-0 parameter equality across arms for the same seed;
  - softmax path bitwise equal to the v3 class;
  - sigmoid weights ≈ 1/T at init;
  - layer-1 placebo: Δz_j = 0 for j ≠ k;
  - forbidden-column guard;
  - balancing touches FIT only.
- Then run a fresh-context reviewer pass (a separate agent with no access to your reasoning) that audits leakage, fold usage and pre-registration compliance. Fix what it finds before you hand over.

Notebooks:

| NB | Purpose | Runs on |
|---|---|---|
| `00_core_tests` | the test suite above, runnable by me on real data in smoke mode | FIT/VAL |
| `01_lecturer_track` | Track A | FIT/VAL |
| `01b_physionet_crosscohort` | Track A2: harmonisation, external validation, attention replication | MIMIC FIT/VAL + PhysioNet A/B |
| `02_arms_train` | Track B training, controls and balanced variants | FIT/VAL |
| `03_correlational_arms` | P1–P4 for both arms, identical pipeline | VAL (DISCOVERY/SCREEN) |
| `04_causal_battery` | P5–P8 | LOC → CONF → EXT |
| `05_interpretable_fix` | Track D | FIT/VAL, then CONF |

## 4 · Track A: lecturer notebook (`01_lecturer_track`)

**From hour 1.** Every ICU hour from hour 1 to the end of the stay gets a prediction for t+1, t+3 and t+6, with the label definition unchanged. Training uses all hours (chunked windows; each row seen once per epoch). For inference on stays longer than the context, use overlapping causal windows so each row is predicted once with ≥ min(t, C) hours of history, or raise `MAX_SEQ_LEN`. Measure the stay-length distribution and choose. Apply this full-coverage view to **all** v4 tracks for comparability, and also report the v3 "most-recent-168 h" view as a bridge to the old numbers. Add an **AUROC/AUPRC-by-ICU-hour** analysis (buckets 1–6, 7–12, 13–24, 25–48, 49–168, >168 h). This is the "how early can it tell" figure the lecturer wants. Draw it three ways: all stays, without early-onset stays (onset < 12 h), and early-onset vs. late-onset as separate curves. In early-onset stays, hours 1–6 are mostly *recognising* sepsis that is already happening or about to; late-onset stays show true *prediction*. Report the two as a separate "recognition vs. prediction" analysis, and never present the hour 1–6 number from all stays as early-prediction evidence on its own.

**Models (each ±ICULOS):**
1. **Our Transformer:** the v3 architecture with v3's winning hyperparameters.
2. **KA-Transformer:** RBF kernel attention, exp(−β‖·‖²) with learned β, as in the paper. Read the exact formulation in the paper: what the distance is computed on, and whether it is normalised. Build two versions:
   - (a) **Paper-faithful**, with the paper's preprocessing, split style, SMOTE, global average pooling and hyperparameters. Note: d_model 512 is not divisible by 6 heads, so pick the nearest valid configuration and log it.
   - (b) **Strict-protocol**, under our rules: causal mask, last-position readout instead of GAP, frozen split, forward-fill and decay only.
3. **LightGBM reference**, ±ICULOS.

**Balancing:** the lecturer requires it. Apply it on FIT only, to each model, as a separate variant:
- SMOTE proper where it is valid: GBDT per-hour features, and the paper-faithful KA if the paper uses flattened windows.
- For sequence models, a sequence-valid method of your choice: oversampling of positive-containing windows, or SMOTE in the frozen trunk's representation space with a head retrain. Justify the choice in one paragraph that I can show the lecturer.
- Recalibrate after balancing (prior shift or Platt fitted on natural-distribution VAL), because balancing distorts the base rate.

**Protocol ladder (main "wow" asset of this track).** Start from paper-faithful KA and change **one** protocol element at a time, measuring each step on the same VAL with stay-level paired-bootstrap CIs. The steps: subject-exclusive chronological split → no backward-fill → causal mask plus last-position readout → our labels, horizons and from-hour-1 coverage → no SMOTE → ±ICULOS. Present it as a waterfall from the paper's reported AUROC (~0.94–0.98) to the strict-protocol number. Frame it as protocol sensitivity, measured, not as an accusation against the authors.

**Metrics:** AUROC and AUPRC per horizon with CIs; calibration (slope and ECE); late-row and no-early-onset strata, as in v3. Also report sensitivity, specificity, F1 and accuracy at a VAL-chosen threshold, so the lecturer can compare with the paper's table. State that accuracy is uninformative at ~1 % prevalence. Include the PhysioNet-2019 utility score as a secondary metric if it's cheap.

## 4b · Track A2: PhysioNet 2019 cross-cohort generalisation (`01b_physionet_crosscohort`)

Goal: show whether the models generalise to other hospitals. This is external validation, and it mirrors the KA paper's own setup (they trained on PhysioNet and tested on MIMIC-IV).

**Harmonise first:**
- Use the **intersection** of MIMIC-IV training features and PhysioNet columns. Drop PhysioNet columns the MIMIC-IV models don't use (e.g. `TroponinI`). MIMIC-only inputs that PhysioNet lacks (`GCS`, `UrineOut_mL`, `PF_ratio`; verify the list) are removed. All cross-cohort models are therefore **reduced-feature models retrained on MIMIC-IV FIT**, and their MIMIC VAL score is reported next to the full-feature model, so the cost of dropping features is visible.
- Static features (Age, Gender, Unit1/Unit2, HospAdmTime) map directly. ICULOS is used only in the ±ICULOS ablation, as everywhere else.
- Verify units and value distributions per feature (e.g. Temp, FiO2 fraction vs. percent) before anything runs. Log every mapping in `Docs/HARMONISATION.md`.
- **Labels:** PhysioNet's `SepsisLabel` is already shifted to 6 h before onset. Recover the onset hour, rebuild the label with the **same definition** as MIMIC (1 iff 0 ≤ HoursToOnset ≤ 6), drop post-onset rows, and report the early-onset share per set.
- Same pipeline as MIMIC: forward-fill and GRU-D decay, prediction from ICU hour 1, horizons t+1/t+3/t+6.
- External data is **never** used to fit normalisation, calibration or thresholds. Apply MIMIC FIT statistics as they are. Report raw and MIMIC-calibrated probabilities, and as a labelled secondary result, recalibration on a small external split.

**Which set is truly external:**
- **setB (Emory) is the clean external test.**
- **setA comes from Beth Israel (BIDMC), the same hospital as MIMIC**, so it is a temporal and same-site check with possible patient overlap. Report it separately and state this caveat explicitly.

**Directions:**
1. MIMIC-IV FIT → PhysioNet setA and setB, separately (primary).
2. PhysioNet → MIMIC-IV VAL (the KA paper's direction). Use a patient-level split inside PhysioNet for training and early stopping.
3. setA ↔ setB, as a cheap site-shift check.

**Models:** our Transformer (reduced features, v3 hyperparameters, primary seed plus the softmax and sigmoid arms' primary seeds), strict-protocol KA, and LightGBM, each ±ICULOS. Balanced variants are not needed here unless they're cheap.

**Report:**
- AUROC and AUPRC with stay-level bootstrap CIs, per direction, set and horizon;
- the drop from internal to external performance;
- calibration-in-the-large and slope;
- AUROC-by-ICU-hour, with early- vs. late-onset curves;
- covariate-shift diagnostics: missingness and measurement-density differences between cohorts. My EDA shows density differs by set and label, so check whether density shortcuts transfer.

**Attention replication (exploratory, labelled):** PhysioNet has no SOFA columns. Build a partial-SOFA severity proxy from available components (MAP, platelets, bilirubin, creatinine; respiratory only if derivable) with the same code on both cohorts. Run the P1–P4 screen with the trained MIMIC models on setB stays. Compare against the same proxy on MIMIC, not against full SOFA. This tests whether the null (or the softmax/sigmoid contrast) holds in another hospital, which makes a strong robustness slide. It never replaces the primary MIMIC analysis.

## 5 · Track B: sigmoid arm (`02_arms_train`, `03_correlational_arms`)

Follow Part D of my PDF exactly:
- **Sigmoid weights:** a_tj = σ(z_tj − log T_t), where T_t is the number of visible keys. No renormalisation. No sparsity or entropy penalties in either arm.
- **Matching:** the same embedding, initialisation seed, optimiser, schedule, batch size, loss, positional encoding, mask, aux head and soup procedure. Assert parameter equality at step 0.
- **Sigmoid-specific settings:** 2× warmup; log mean total mass M_t every epoch; clip gradients. If the sigmoid arm fails the gate, tune **learning rate only**, report it as a limitation, and also report the untuned run.
- **Retrain softmax from scratch alongside it** in the same core, with **5 paired seeds per arm**. The v3 checkpoint serves only as a reproducibility check.
- **Analyse both** the checkpoint soup and the best single epoch for both arms. The soup could blur head specialisation.
- **Controls, identical for both arms:** untrained (same init), shuffled-label (v3 protocol as primary, plus a **budget-matched** version as robustness), circular-shift, donor and random-projection severity. Use the **same DISCOVERY/SCREEN stay IDs** for both arms.
- **DVs:** clr(α) for softmax; log a for sigmoid, uncentred, with a query-position covariate. Report Δz-based results in both arms.
- **Extra analyses:**
  - Report per-head 95 % **prediction intervals** (τ), not just pooled CIs.
  - Run the softmax floor-effect check: fraction of α < 1e-6.
  - Run reversed residualisation (severity residualised on deviation) as a clearly labelled exploratory sensitivity analysis.
  - Place the result in the PDF's four-outcome table (softmax × sigmoid).
- **Balanced arms:** one seed each with the best hyperparameters, FIT-only balancing, reported in an appendix. If it's cheap, include one line on whether balancing changes |b1| relative to controls.
- Optional, low priority: rerun v3 ladder rungs C1, C3 and C4 with 3 seeds, so the lever claims are defensible.

## 6 · Track C: rescue without cheating

**The gate.** Propose a replacement for the 0.85 AUROC gate, for example: AUROC CI lower bound > GBDT reference **and** AUPRC lift ≥ 3× base rate at every horizon. **Ask me before adopting it**, and log it in `Docs/DEVIATIONS.md`.

**The causal stage proceeds.** Rung 0 (natural severity sensitivity, P1–P4) and rung 1 (allocation by normalisation, M0 via P7 plus the sigmoid contrast) are separate questions in my PDF. A rung-0 null does not block rung 1. Check what v3 pre-registered for head carry-forward (`CORR_TOP_K = 6`). Run the model-level interventions that need no head selection (S0/S1 input splices, S4 logit δ, S6 query scaling), and carry the pre-specified top-k forward, explicitly labelled "carried forward without passing the screen". Keep the exact nulls (common shift, placebo j < k, layer-1 zero), random-network and shuffled controls, and DAS only with all three of its controls. In the sigmoid arm, measure learned common shift (lateral inhibition) and the total-mass budget.

**Storylines.** Write `Docs/STORY.md` with ranked storylines. Each one is triggered by a result, has one headline claim, one hero figure, and a checklist that must pass before it is claimed. Candidates (add better ones if the data suggest them):
- **S1 · "Attention heatmaps in a clinical Transformer fail a sanity check."** Trigger: trained ≈ untrained ≈ shuffled. Mechanism: SOFA's ingredients are inputs and attention logits are bilinear in inputs, so any network co-varies with severity; the small surrogate controls confirm this. This fits the attention-as-explanation debate (Jain & Wallace; Wiegreffe & Pinter; Adebayo sanity checks) in a high-stakes clinical setting.
- **S2 · "The normaliser decides."** Whichever cell of the four-outcome table occurs, with its check passed.
- **S3 · "A network never forced to budget learns one."** Trigger: sigmoid total mass roughly constant, or a learned common shift.
- **S4 · KA-Transformer protocol ladder.** The waterfall from Track A.
- **S5 · Heterogeneity.** Trigger: I² ≈ 0.99. Exploratory subgroup search (e.g. septic vs. non-septic stays, unit, measurement density, acuity) on DISCOVERY with FDR control, confirmed on SCREEN.
- **S5b · Cross-hospital.** Does performance, and the attention finding, survive on Emory (setB)? How large is the external drop compared with the KA paper's reported MIMIC result?
- **S6 · Diagnose, then fix.** The Track D model passes the sanity checks that the standard model fails, at non-inferior AUROC.

**Presentation and paper.** Produce a consistent figure set (publication style, colour-blind safe) for the triggered storylines, plus `Docs/PAPER_SKELETON.md`: title options, abstract, 3 contributions, figure list, and a limitations section. Target an ML-for-health venue (ML4H, CHIL or MLHC style). Also produce `Docs/SLIDES_OUTLINE.md` with 10–12 slides and the hero figure per slide.

## 7 · Track D: interpretable-by-design fix (`05_interpretable_fix`)

Choose candidates **from the results** using this decision table, and pre-register the choice before training:

| Observed | Implication | Candidate fix |
|---|---|---|
| Coupling is architectural (trained ≈ untrained) | Hour-level α can't separate what the model *uses* | Variable-level tokens (STraTS-style observation tokens, so attention is over named clinical variables), or KA-style RBF attention ("attended because similar"). Then re-run the sanity checks |
| Softmax floor effect / sigmoid differs | The zero-sum constraint distorts α | Sigmoid or entmax-1.5 normaliser as the explanation layer |
| Causal stage supports M0 | α is mostly normalisation | Explain with Δz, norm-weighted attention (Kobayashi) or ALTI, not raw α |
| Huge per-stay heterogeneity | Explanations are unstable | Report explanations with seed and bootstrap stability bands |

Constraints:
- SOFA columns may not be targets or inputs, so no SOFA concept bottleneck; that would make the severity analysis circular. Concepts built from raw vitals are fine.

**Scorecard** (thresholds pre-registered; winner chosen by score, not by eye):
1. AUROC non-inferior to the matched softmax model (margin 0.01, paired bootstrap);
2. its explanation differs from the untrained and shuffled controls beyond the band (Adebayo-style);
3. faithfulness: deletion/insertion AUC and comprehensiveness/sufficiency better than random and better than raw α;
4. stability across 5 seeds.

Compare these explanation methods on every candidate: raw α, clr, norm-weighted, rollout (softmax only), Integrated Gradients, gradient×input.

## 8 · Pre-registration and statistics

- Before `03`–`05` touch real data, write `Docs/PREREGISTRATION.md` and timestamp it in the folder. It must contain: all thresholds, SESOIs, k, margins, τ thresholds, storyline triggers and the Track D scorecard. Freeze it. Every later change goes in `Docs/DEVIATIONS.md` with the reason and date.
- Unit of inference is the stay, or the subject when a subject has several stays. Use stay-level paired bootstrap for model comparisons and seed-level variance for any training claim; a lever effect smaller than the seed SD is "not resolved". Report effect sizes with CIs; p-values are secondary. Use TOST for nulls, and FDR across sensitivity families.

## 9 · How we work

1. **Phase 0 (no code yet):** read everything and send me a one-page plan covering open questions, risks you see in my design, and any place where my PDF and v3 disagree. Wait for my reply.
2. **Phase 1:** core plus `00`, all tests green on synthetic data, reviewer pass done, then hand over.
3. **Phase 2:** `01`, `01b` and `02`. I run them and give you the aggregate outputs.
4. **Phase 3:** freeze the pre-registration, then `03` and `04`.
5. **Phase 4:** `STORY.md`, figures, paper skeleton, slides outline.
6. **Phase 5:** `05` and the final scorecard.

Between phases, give me a short status: what ran, what it showed, and what changed in the plan. Ask me before:
- adopting the new gate;
- any deviation from pre-registered rules;
- KA-paper ambiguities that change results;
- writing outside `Notebooks/v4/` and `Docs/`.

Otherwise make the call and log it.

**Definition of done:** every notebook runs top to bottom in smoke mode and in full mode; every number in `STORY.md` traces to a logged run; every claim names the check that licenses it; and a null is reported as a null at the right scope ("this model does not show X"), never hidden.

## 10 · References

"Core" = in `Resources/` from the first run. The other phases list when a paper is needed; ask me to add it then. The methodology PDF (`attention_method_8_points.pdf`) has a longer list for the P1–P8 details. If a citation there conflicts with this list, flag it.

| # | Reference | Used for | Phase |
|---|---|---|---|
| R1 | Unknown authors (2025). Application of the KA-Transformer model to early sepsis prediction: a hybrid network analysis based on time series data. *Discover Applied Sciences*. doi:10.1007/s42452-025-06628-8 | Track A reference model, protocol ladder | Core |
| R2 | Jain, S., & Wallace, B. C. (2019). Attention is not Explanation. *NAACL-HLT 2019*. | Attention-as-explanation debate | Core |
| R3 | Wiegreffe, S., & Pinter, Y. (2019). Attention is not not Explanation. *EMNLP-IJCNLP 2019*. | Same debate, counter-position | Core |
| R4 | Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity Checks for Saliency Maps. *NeurIPS 31*. | Untrained and shuffled-label controls, storyline S1 | Core |
| R5 | Ramapuram, J., et al. (2025). Theory, Analysis, and Best Practices for Sigmoid Self-Attention. *ICLR 2025*. | Sigmoid arm | Core |
| R6 | Singer, M., et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*, 315(8), 801–810. | Label definition | Core |
| R7 | Reyna, M. A., et al. (2020). Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*, 48(2), 210–217. | PhysioNet data, label shift, utility score | Core |
| R8 | Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018). Recurrent Neural Networks for Multivariate Time Series with Missing Values. *Scientific Reports*, 8, 6085. | GRU-D decay inputs | Core |
| R9 | Johnson, A. E. W., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10, 1. | Cohort | 4 |
| R10 | Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357. | Lecturer balancing | 2 |
| R11 | Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*. | Loss | 4 |
| R12 | Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 30*. | GBDT reference | 4 |
| R13 | Aitchison, J. (1982). The Statistical Analysis of Compositional Data. *JRSS B*, 44(2), 139–177. | clr DV | 3 |
| R14 | DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*, 7(3), 177–188. | Stage-2 random-effects meta-analysis | 3 |
| R15 | Lakens, D. (2017). Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses. *Social Psychological and Personality Science*, 8(4), 355–362. | TOST | 3 |
| R16 | Kobayashi, G., Kuribayashi, T., Yokoi, S., & Inui, K. (2020). Attention is Not Only a Weight: Analyzing Transformers with Vector Norms. *EMNLP 2020*. | Norm-weighted DV, Track D | 3 |
| R17 | Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Farmer, J. D. (1992). Testing for nonlinearity in time series: the method of surrogate data. *Physica D*, 58, 77–94. | Circular-shift control | 3 |
| R18 | Geiger, A., Lu, H., Icard, T., & Potts, C. (2021). Causal Abstractions of Neural Networks. *NeurIPS 34*. | IIA, P7 | 3 |
| R19 | Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. (2024). Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations. *CLeaR 2024*. | DAS, P8 | 3 |
| R20 | Makelov, A., Lange, G., & Nanda, N. (2024). Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching. *ICLR 2024*. | DAS controls | 3 |
| R21 | Heimersheim, S., & Nanda, N. (2024). How to use and interpret activation patching. *arXiv:2404.15255*. | Patching protocol, P5 | 3 |
| R22 | Sutter, D., Minder, J., Hofmann, T., & Pimentel, T. (2025). The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability? *arXiv* (verify title and venue). | Linear-only alignment, random-network control | 3 |
| R23 | Vig, J., et al. (2020). Investigating Gender Bias in Language Models Using Causal Mediation Analysis. *NeurIPS 33*. | Real-pair counterfactuals | 3 |
| R24 | Mather, M., & Sutherland, M. R. (2011). Arousal-Biased Competition in Perception and Memory. *Perspectives on Psychological Science*, 6(2), 114–133. | M2, ABC framing | 3 |
| R25 | Reynolds, J. H., & Heeger, D. J. (2009). The Normalization Model of Attention. *Neuron*, 61(2), 168–185. | M0, normalisation framing | 3 |
| R26 | Tipirneni, S., & Reddy, C. K. (2022). Self-Supervised Transformer for Sparse and Irregularly Sampled Multivariate Clinical Time-Series. *ACM TKDD*, 16(6). | STraTS-style tokens, Track D | 5 |
| R27 | Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML 2017*. | Integrated Gradients | 5 |
| R28 | Peters, B., Niculae, V., & Martins, A. F. T. (2019). Sparse Sequence-to-Sequence Models. *ACL 2019*. | entmax-1.5 | 5 |
| R29 | DeYoung, J., et al. (2020). ERASER: A Benchmark to Evaluate Rationalized NLP Models. *ACL 2020*. | Comprehensiveness and sufficiency | 5 |
| R30 | Abnar, S., & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. *ACL 2020*. | Attention rollout baseline | 5 |
| R31 | Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *PNAS*, 115(11), 2600–2606. | Pre-registration framing | 4 |

Phases: 2 = Track A/A2/B training, 3 = analysis notebooks `03`/`04`, 4 = story and paper, 5 = Track D. Before the paper skeleton is final, check every author list, year and venue against the PDFs; some entries above are written from memory.
