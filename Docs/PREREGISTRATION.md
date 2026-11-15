# Pre-registration: does attention track severity? (v4, Phase 3–5)

**Status: FROZEN 2026-11-15.** Drafted by Claude, reviewed and approved by Miaki on 2026-11-15 (section 13). Stamp: `Docs/PREREGISTRATION.frozen`. From here on this file is not edited; changes go in `Docs/DEVIATIONS.md` (section 0).

Rules in force together with this file: `Resources/attention_method_8_points.pdf` ("the PDF"), `Docs/FABLE_PROMPT_sepsis_v4.md` ("the brief"), `Docs/DECISIONS.md`, `Docs/DEVIATIONS.md`. Where this file is more specific than the PDF, this file decides. Where it fills a gap or tightens a PDF default, the line is marked **[new]** and listed in section 13, with Miaki's decision.

## 0 · Freezing and changing this file

1. Miaki reviews, edits and approves section 13.
2. The status line becomes `FROZEN <date>`, the file is committed, and `Docs/PREREGISTRATION.frozen` is written with the date, the approver and the sha256 of this file. That stamp is what `data.FoldGuard` checks before LOC, CONF or EXT can be loaded.
3. After freezing, this file is not edited. Every change goes in `Docs/DEVIATIONS.md` with date, reason and approval, before the affected analysis is run. Results obtained before a deviation are kept and shown beside the new ones.
4. Rules that hold throughout (brief, non-negotiable 5): no control band is moved, no control is dropped, no DV is changed and no heads are re-picked after results are seen. Anything not specified here is exploratory and is labelled so.

## 1 · What was already known when this was written

Stated so a reader can judge which thresholds could have been influenced by results.

| Seen | Not seen |
|---|---|
| v3 softmax screen on the v3 checkpoint (seed 42): band 0.3458, largest trained effect −0.268 (L4H3), 0 of 32 heads pass, 7 equivalent to zero. Only the top 12 rows of each head table were displayed. | Any attention capture, regression or intervention on a v4 model |
| All Phase 2 VAL performance numbers (`01`, `01b`, `02`): both arms pass the gate; sigmoid − softmax = −0.0071 (SD 0.0038 over 5 paired seeds) | Anything from LOC, CONF or EXT. These folds have never been loaded |
| The per-epoch mass probe: sigmoid 0.24–0.26, `kernel_raw` 0.039. **This probe reads the first two evaluation batches, which are the 256 shortest VAL stays, at every query position.** It is not the analysis population of section 5 | Total mass on the analysis queries (position ≥ 8), by layer or by position |
| Test 6 on small smoke models: in layers ≥ 2 the logit change for keys before the intervened hour is 0.014–0.028, not 0 | The size of any intervention effect on a trained v4 model |

Consequence: the correlational thresholds (band P95, SESOI 0.05, top-k 6) are v3's and were fixed before any v4 model existed. The sigmoid mass floor in section 6 was chosen after the probe value above was seen; for that reason the deciding criterion there is a functional test whose result is not yet known. The outcome "attention contributes little in either arm" (6.2) was added because `kernel_raw` reached 0.796 AUROC with near-zero probe mass.

## 2 · Questions and the claim ladder

Each rung is reported on its own. A null on one rung does not block the next (PDF Part C).

| Rung | Claim | Decided by | Section |
|---|---|---|---|
| 0 | Attention is severity-sensitive beyond what control networks and control inputs show | P1–P4, `03` | 5 |
| 1 | The network allocates by normalisation: raising one hour's priority raises its logit and leaves the others in proportion (model M0) | P5–P7, `04`, plus the sigmoid contrast | 7 |
| 2 | Sharpness depends on context (M1) | selectivity slope under do(Priority), IIA M1 > M0 | 7 |
| 3 | Sharpness is gated by arousal (M2) | selectivity slope under do(Arousal), interaction, IIA M2 > M0 and M1 | 7 |

Scope of every conclusion: "this model, on this cohort, does or does not show X". Never "ABC is true/false".

## 3 · Data, folds and models

**Cohort and view.** MIMIC-IV v3.1 hourly cohort, 74,704 stays; label = 1 iff 0 ≤ HoursToOnset ≤ 6; post-onset rows dropped; horizons t+1, t+3, t+6. Each stay contributes its most recent 168 hours (`COVERAGE = "last"`, `MAX_SEQ_LEN = 168`, decision C25). Frozen split sha1 `65441cff1a47`, subjects per fold: FIT 33,057 · VAL 5,880 · LOC 6,340 · CONF 8,227 · EXT 2,571.

**Models analysed** (all trained in Phase 2, run folder `v4_last168`; nothing is retrained for the primary analysis):

| Role | Models | Use |
|---|---|---|
| Primary arms | `softmax_s{42,1042,2042,3042,4042}`, `sigmoid_s{42,1042,2042,3042,4042}` | Every confirmatory analysis |
| Checkpoint | Soup (`best.pt`) is primary. Best single epoch (`best_epoch.pt`) is a pre-specified sensitivity analysis | The soup is the reported, calibrated model |
| Untrained control | Each run's own step-0 weights (`init.pt`) | Band member (C9) |
| Shuffled-label, v3 protocol | `softmax_shuf_v3`, `sigmoid_shuf_v3` (one per arm, used for all five seeds of that arm) | Band member (A2) |
| Shuffled-label, budget-matched; budget-matched aux-off | `{arm}_shuf_budget`, `{arm}_shuf_budget_noaux` | Reported beside the band, never in it (A2) |
| Exploratory | `kernel_s{42,1042,2042}`, `kernel_raw_s42`, `kernel_shuf_v3`; reduced-feature `r33_*` models for PhysioNet | Section 12 only |
| Excluded | Balanced models | No interpretability analysis (brief, non-negotiable 3) |

Every reported run pins `REPO_REF` to a commit hash, and that hash is logged with the result.

**Gate (DEVIATIONS D1), already evaluated.** At the primary seed both arms pass at every horizon: softmax +0.030 / +0.042 / +0.048 AUROC over LightGBM (all lower bounds > 0), AUPRC lift ×5.7–5.9; sigmoid +0.022 / +0.033 / +0.044, lift ×5.1–5.2. The sigmoid fallback (A9: learning-rate-only tuning plus a LayerScale pair in both arms, the unmodified pair staying primary) was pre-registered and was **not triggered**. It stays in force only if a later deviation requires retraining.

**Fold use.**

| Notebook | Folds | Rule |
|---|---|---|
| `03` | FIT, VAL | DISCOVERY and SCREEN are subsets of VAL |
| `04` | LOC → CONF-A | LOC: exploration, fitting, every choice. Then `Docs/LOC_CHOICES.md` is written and committed. Only then is CONF-A loaded, once, with every choice fixed |
| `05` | FIT, VAL → CONF-B | Candidates trained and selected on FIT/VAL; scorecard confirmed once on CONF-B |
| Final table | EXT | Loaded exactly once, after `04` and `05` are complete and committed (A14). Contents fixed here: (i) performance of the frozen models, (ii) replication of the CONF-A confirmatory quantities and IIA with every alignment frozen (PDF P7) **[new: reconciles A14 with the PDF]** |

**CONF-A / CONF-B (A6).** Split by subject with a rule that needs no data: a CONF subject is in CONF-A iff `int(sha1(f"{subject_id}|CONF|65441cff1a47").hexdigest(), 16) % 2 == 0`, otherwise CONF-B **[new]**. The two counts are written into the freeze stamp the first time the split table is read. The rule is enforced in `data.FoldGuard` before `04` is handed over.

## 4 · Constants

| Constant | Value | Origin |
|---|---|---|
| Severity | `SOFA_total`, forward-filled within stay; analysis only, never a model input | v3 |
| DISCOVERY / SCREEN | v3's 800 + 800 VAL stays, disjoint subjects, IDs loaded from v3's `correlation/correlational_summary.json`, never re-drawn | v3 |
| Minimum stay length; query positions; queries per stay | 24 h; position ≥ 8 (`CORR_MIN_T`); up to 24, drawn with v3's per-stay seed | v3, A11 |
| SESOI (standardised effects) | 0.05 | v3 |
| Control band | 95th percentile of \|pooled control effect\| | v3 |
| Decision threshold | max(band, SESOI) | v3 |
| Heads carried forward | top 6 by \|b1\| on DISCOVERY (A3) | v3 `CORR_TOP_K`, PDF |
| Severity classes for pairs | severe ≥ Q90, routine ≤ Q50 of `SOFA_total` on FIT rows | PDF |
| Pairs | about 2,000 base triples per fold, at most 3 per stay | PDF |
| IIA match threshold | Kendall τ ≥ 0.3 | PDF default |
| IIA margin; DAS margin | 0.05 (absolute), CI excluding 0 | **[new]** |
| Non-inferiority margin (Track D) | 0.01 AUROC | brief |
| Seeds | 42, 1042, 2042, 3042, 4042, paired across arms | Phase 0 |
| Confidence level | 95 %; bootstrap resamples subjects | brief §8 |

## 5 · Correlational stage (`03`, P1–P4)

One pipeline, run unchanged on every model. Notation: t = query hour, j = key hour (j ≤ t), z = pre-normaliser logit, α or a = attention weight.

**5.1 Dependent variables.** Capture in fp32 with dropout off.

| DV | Definition | Arm | Role |
|---|---|---|---|
| clr | z_tj − mean over visible keys of z_ti (equals clr(α) exactly) | softmax | Primary screening DV (v3) |
| log a | log σ(z_tj − log T_t), uncentred | sigmoid | Primary screening DV (PDF Part D) |
| row-centred logit | z_tj − mean over visible keys of z_ti, computed the same way in both arms | both | **Primary cross-arm quantity in `03`** (A11). In the softmax arm it is the clr DV. Δz proper needs an intervention and belongs to `04` (PDF P1) |
| norm-weighted | log(weight × ‖v_j W_O‖), row-centred | both | Secondary (P2). If it disagrees with the primary DV, both are reported |

**5.2 Stage 1, per stay and head** (v3 design, unchanged): DV_tj = b0 + b1·severity_j + b2·log(t−j+1) + b3·log(t+1) + b4·density_j + b5·deviation_j, both sides standardised within stay, ordinary least squares. deviation_j is residualised on severity within stay. b3 is the query-position covariate the sigmoid DV requires; it is in the design for both arms. ICULOS_t is not added (collinear with position within a stay).

**5.3 Stage 2** (v3 estimator, the primary decision rule, A12): DerSimonian–Laird random-effects pooling of b1 across stays, 95 % CI; beside it the plain mean with a bootstrap CI.

**5.4 Controls and the band.** For each analysed model (arm, seed, checkpoint) and each DV, the band is the 95th percentile of |pooled b1| over all heads of five control sets, all on DISCOVERY:

1. untrained: the model's own `init.pt`, real severity;
2. shuffled-label, v3 protocol: the arm's `*_shuf_v3` model, real severity;
3. the trained model with circular-shift severity;
4. the trained model with donor severity (another stay);
5. the trained model with random-projection severity.

Difference from v3, fixed here before any v4 result: v3's untrained control used a different seed; v4 uses the trained model's own step-0 weights (C9). The budget-matched and aux-off shuffled-label controls are run through the same pipeline and reported beside the band. They never enter it.

Limitation, stated in advance: the v3-protocol shuffled-label control is one trained model per arm and serves all five seeds of that arm, so the variance of this control over training seeds is not estimated. The band and the count n_shuf (5.6) carry that unquantified uncertainty. The untrained control does vary by seed. This goes into the paper's limitations section.

**5.5 Head-level rule** (v3, unchanged). A head *passes on DISCOVERY* iff |b1| > max(band, 0.05) and both CIs exclude 0. It is *confirmed on SCREEN* iff b1 has the same sign, |b1| > 0.05 and its CI excludes 0.

**5.6 Seed-level rule (A10) [new].** Head indices are not aligned across seeds, so the decision uses counts. The band is a percentile of a pool that contains the control networks, so some control heads lie above it by construction; one trained head above the band is therefore not evidence by itself. For each seed s:

- n_trained = number of trained heads that pass on DISCOVERY and are confirmed on SCREEN;
- n_init, n_shuf = the same count for the untrained network and for the v3-protocol shuffled-label network, by the same rule and threshold;
- excess E_s = n_trained − max(n_init, n_shuf).

| Verdict for an arm | Rule |
|---|---|
| PRESENT | E_s > 0 in 5 of 5 seeds (exact one-sided sign test, p = 1/32 = 0.031) |
| ABSENT | E_s ≤ 0 in at least 4 of 5 seeds |
| SEED-DEPENDENT | anything else; reported as "not established", no claim either way |

The same three-way rule is used for every head-level quantity in section 7.

**5.7 Reading of the result (A2).**

- ABSENT against the primary band: *"sepsis labels add nothing detectable beyond physiology learning."* The v3-protocol control has the auxiliary forecasting head, so this is the narrower claim.
- The broader claim (*nothing beyond architecture and inputs*) is made only if the verdict is also ABSENT when the aux-off shuffled-label control replaces the v3-protocol one in E_s.
- Trained heads are expected to differ from zero (v3: the 12 largest effects all replicate). The null here is "not distinguishable from controls", not "no association". Per-head equivalence to zero (TOST against ±0.05, v3 rule) is reported as a count and is not the decision rule.

**5.8 Heads carried forward (A3).** For each v4 model, the 6 heads with the largest |b1| on DISCOVERY, whether or not they pass, labelled "carried forward without passing the screen" when they do not. Layer-1 heads are reported but flagged: they cannot show effects on other hours.

**5.9 Heterogeneity (A12).** Any statement about I², τ or prediction intervals uses stage-1 errors that are cluster-robust on the key hour within stay (severity_j repeats across up to 24 queries), with DerSimonian–Laird on those errors, and a 95 % prediction interval per head. The v3 estimator remains the decision rule for the band.

**5.10 Pre-specified sensitivity analyses.** None changes a primary verdict. Benjamini–Hochberg FDR at q = 0.05 within each family.

| Family | Members |
|---|---|
| Severity definition (PDF Part C) | each SOFA component (resp, coag, liver, cv, cns, renal) and `SOFA_delta`, primary seed, both arms |
| Specification | norm-weighted DV; best single epoch instead of soup; stays longer than 168 h excluded (they are truncated); queries at position ≥ 24 only; cluster-robust inference |

**5.11 Always reported.** VIF of the design (above 5 requires a re-specification, logged as a deviation, before `04`); the fraction of softmax weights below 1e-6 among analysed pairs, per head (floor-effect check; above 5 % in a head is flagged); total mass by query position (section 6).

## 6 · Is the sigmoid arm alive? Total mass

**6.1 Measurement.** M_t = Σ_{j ≤ t} a_tj per head, on the DISCOVERY and SCREEN stays, soup checkpoint, reported per layer and by query position, and compared with the 1/(1+T) arithmetic at step 0 (`init.pt`) (A11). The headline value is the mean over analysis queries (position ≥ 8) per layer. The Phase 2 probe value (0.24–0.26) is not used: it came from the 256 shortest VAL stays.

**6.2 "Collapsed", and "attention contributes little in either arm" [new].** The PDF requires a sigmoid arm that is "not dead" before it can serve in the four-outcome table. Two measurements are taken at every seed, on both arms:

- *Functional (knockout):* the output of every attention sublayer is set to zero, the residual stream kept. The knockout **matters** for a model iff it lowers VAL AUROC (mean of horizons) by at least 0.01 with the stay-level paired-bootstrap 95 % CI excluding 0. Otherwise it **costs little** (a drop below 0.01, or a CI that includes 0).
- *Mass floor (sigmoid only):* mean M_t over analysis queries is at least 0.05 in at least one layer.

Outcome per seed, from the paired sigmoid and softmax models:

| Sigmoid knockout | Softmax knockout | Seed outcome |
|---|---|---|
| matters, and the mass floor is met | either | sigmoid **alive** |
| costs little | costs little | **attention contributes little in either arm** |
| costs little | matters | sigmoid **collapsed** (attention bypassed in the sigmoid arm only) |
| matters, but the mass floor is not met | either | sigmoid **collapsed** |

Outcome for the study, by count over the five seeds:

| Study outcome | Rule | How the four-outcome table (section 8) is then read |
|---|---|---|
| Sigmoid ALIVE | at least 4 of 5 seeds alive | As written in section 8 |
| ATTENTION CONTRIBUTES LITTLE IN EITHER ARM | at least 4 of 5 seeds have that seed outcome | Its own pre-specified outcome, of the S1 type. The sigmoid arm is **not** declared collapsed and no difference between the normalisers is read into the knockout. No cell of the table is claimed. The verdicts on λ1 and c are still computed and shown in the table layout for both arms, each labelled "pattern in logits that the prediction does not depend on". Rungs 1–3 are then statements about the attention logits only, not about how the model reaches its prediction. Headline: neither normaliser's attention map can explain a prediction that barely uses attention |
| Sigmoid NOT ALIVE | neither of the above | The sigmoid arm cannot serve as a null. No cell is claimed; the softmax verdict is reported alone, with the per-seed breakdown of the sigmoid outcomes |

Setting attention outputs to zero takes the network off its training distribution, which tends to enlarge the drop, so "costs little" under this knockout is the conservative direction for that verdict. A second knockout, weights replaced by a uniform row with the same total mass, is reported beside it to separate "attention is used" from "the pattern is used"; it is descriptive and decides nothing. The same two knockouts on `kernel_raw` are exploratory (section 12).

**6.3 Scale drift.** Flagged if mean M_t exceeds 5 in any layer, or if the logged per-epoch mass at the last epoch is more than twice its value three epochs earlier.

**6.4 Does mass depend on what is in the window? (PDF Part D)** Two-stage regression as in 5.2–5.3 with DV M_t and covariates: mean severity of the visible keys, log T_t, mean measurement density. "Insensitive to composition" iff the standardised severity coefficient is equivalent to zero (TOST, ±0.05) under the seed rule applied at model level (5 of 5 seeds). This is the natural-data half of storyline S3; the interventional half is in 7.5.

## 7 · Causal stage (`04`, P5–P8)

Order: LOC → `LOC_CHOICES.md` committed → CONF-A once → EXT once at the very end (section 3). All five seeds of both arms, soup checkpoint; best single epoch at the primary seed as sensitivity. Notation: k = intervened hour, t = readout (query) hour, S = {j : k < j < t}.

**7.1 Pairs (do(Priority_k)).**

- Base: a stay from the fold, an hour k ≥ 3 whose severity class is routine, and a readout t ≥ max(8, k + 6), so S holds at least 5 keys **[new: the PDF default is t − k ≥ 3]**.
- Severe donor: an hour of class severe from another subject in the same fold, matched on ICULOS (±4 h), number of measured features (±2), Unit1/Unit2 pattern, and previous-hour HR, MAP, Resp and O2Sat each within 1 FIT standard deviation of the base's previous hour **[new: the tolerance]**.
- **Sham donor [new]:** a second donor matched in exactly the same way but of class *routine*, like the base hour. It carries the disturbance of a splice without a change in severity.
- The effect of interest is always the paired contrast *severe splice − sham splice* on the same (stay, k, t). Bases without both donors are dropped and the yield is reported.
- Secondary: "down" pairs (severe base, routine donor). Plausibility (PDF): frequency of one-hour excursions of the spliced size in raw stays; loosely matched versus context-matched donors.

**7.2 Sites.**

| Site | What | Status |
|---|---|---|
| S1 | The embedding of hour k is replaced by the donor hour's embedding. No other hour's input changes | **Primary** for every allocation quantity below |
| S0 | Raw inputs of hour k replaced, decay features recomputed downstream. Later hours' inputs change too | Reported beside S1; a difference between S0 and S1 is itself a result |
| S4 | δ = +2 added to logit z_tk before the normaliser, one layer at a time | Model-level: does the output depend on allocation? |
| S6 | Query vector at (layer ≥ 2, t) scaled by 0.5 and by 2 | Reference: what a pure gain change looks like. Not evidence for M2 |
| S2, S3, S5 | Residual sweep, head/MLP output, value vector | Exploratory, LOC only |

**7.3 Exact checks (replaces the PDF's "placebo set j < k is exactly zero"; open item O1).** These must hold to float precision in both arms. A violation is a bug and stops the run.

| # | Statement | Holds for |
|---|---|---|
| E1 | Two identical runs give Δz = 0 | every site |
| E2 | Δz_tj = 0 for every query t < k, every key, every layer (causality) | every site |
| E3 | In layer 1, Δz_tj = 0 for every t ≠ k and j ≠ k | S1 only. Under S0 the inputs after k change, so E3 does not apply |
| E4 | Under do(Arousal_t) (7.6), model outputs at hours < t are unchanged | arousal intervention |

What is **not** exact, and how it is used instead:

- In layers ≥ 2, for queries t > k, Δz_tj is not zero for keys j < k: the query has already read hour k. The key side of those keys is unchanged in every layer, so the set {j < k} isolates the **query-side** change. It is reported as a reference set (its shift and slope), never as a noise floor.
- The empirical noise floor is the **sham splice**: every statistic is the severe − sham contrast, and a sham − sham contrast (two different routine donors) on the trained model is a member of the control band.

**7.4 Quantities.** Logit-scale quantities are divided by the standard deviation of z over the visible keys of the same query row in the base run, so that the SESOI of 0.05 applies **[new: this standardisation]**. Each is computed per head in layers ≥ 2 (S1) as the mean severe − sham contrast, with a subject-level bootstrap CI.

| Symbol | Definition | Null |
|---|---|---|
| d (direct effect) | Δz_tk, the change in the logit to the intervened hour | 0. This is the manipulation check: without it nothing else is interpretable |
| c (common shift) | mean over S of Δz_tj | softmax: any value (the normaliser removes it). **sigmoid: 0** (independent gates); a negative c is learned inhibition |
| λ1 (selectivity slope) | slope of r_j = Δz_tj − mean_S Δz on severity_j standardised within S. Positive = low-severity hours fall further (PDF sign convention) | 0 in both arms (M0) |
| ΔM, κ (sigmoid only) | change in total mass M_t; compensation κ = −Σ_{j≠k} Δa_tj / Δa_tk | κ = 0 independent gates; κ = 1 full compensation (a budget) |
| output effect | change in the t+1 logit divided by the SD of that logit over the fold's rows | 0 |

**7.5 Decision rule for d, c and λ1.** Control band = 95th percentile over heads of the pooled |statistic| from: the untrained network (`init.pt`), the v3-protocol shuffled-label network, the sham − sham contrast on the trained network, and, for λ1, the trained network with the three surrogate severities of 5.4. A head passes on LOC iff |statistic| > max(band, 0.05) with CI excluding 0; it is confirmed on CONF-A iff same sign, |statistic| > 0.05, CI excluding 0. Verdict per arm by the seed rule of 5.6 (counts against the two control networks). The six carried-forward heads are also reported by name.

Sigmoid budget (interventional half of S3): κ pooled over heads in layers ≥ 2 as a ratio of means. "Learned budget" iff its 95 % CI lies inside [0.5, 1.5]; "independent gates" iff inside [−0.25, 0.25]; anything else is reported as partial compensation with the number **[new]**. Model-level, so the verdict needs 5 of 5 seeds.

**7.6 do(Arousal_t).** Arousal index = z(HR) + z(Resp) at the readout hour from the decayed input, FIT statistics (PDF). Intervention: HR and Resp at hour t replaced by those of a matched donor hour with index ≥ Q90 (base index ≤ Q50); sham donor with index ≤ Q50. Keys j < t are untouched in every layer, so the effect is purely query-side. Quantities: λ1 over all j < t, and the change in row entropy (softmax). Also reported: the interaction (d against the natural arousal tertile of hour t), a dose-response across arousal quantiles, and the placebo index (the same index circularly shifted within stay), which must not gate sharpness. `SOFA_total` at t as the alternative index is a sensitivity analysis.

**7.7 High-level models and IIA (P6, P7).** Fitted on LOC from the model's own logits; CONF-A untouched until then. log P_j is linear in severity_j per head (isotonic as sensitivity). Only changes in sharpness are tested, never an absolute γ (PDF B0a).

| Model | Prediction for r_j on S under do(Priority_k) | Under do(Arousal_t) |
|---|---|---|
| M0 fixed γ | constant (no structure) | no change |
| M1 saliency gain | Δγ·(log P_j − mean), Δγ > 0 when k becomes the strongest item | no change |
| M2 arousal gain | constant, as M0 | Δγ·(log P_j − mean), Δγ ∝ change in arousal |

- Observed vector: Δ log α over S (softmax), Δz over S (sigmoid).
- Match when the model predicts structure: Kendall τ(predicted, observed) ≥ 0.3 (PDF).
- **Match when the model predicts a constant vector [new]:** the PDF's τ is undefined there. The pair matches iff |τ(log P_j, observed)| < 0.3, that is, no rank structure at the same threshold, and d has the predicted sign.
- IIA = share of pairs matched. Reported per model and separately for priority pairs, arousal pairs and sham pairs (where every model predicts no change and a network that ignores interventions scores high).
- Floor: the same computation with log P permuted within S.
- "Mi beats the floor" and "Mi is supported over Mj": difference in IIA ≥ 0.05 with a subject-level bootstrap CI excluding 0, on CONF-A, and the same sign on EXT with everything frozen. IIA_risk (direction and tertile of the t+1 logit change) is secondary.

**7.8 DAS (P8).** Run only if no hand-specified model beats the floor. Linear orthogonal R only, trained on LOC, evaluated on CONF-A, primary seed, subspace sizes m ∈ {1, 2, 4, 8, 16}. It counts only if held-out IIA exceeds **all three** controls (randomly initialised network, shuffled counterfactual labels, random subspace of the same size) by at least 0.05, and the subspace patch agrees with the whole-component patch (illusion check). If it does not, the alignment is reported as absent, which is evidence against the hypothesis (PDF).

**7.9 Decisions per rung** (per arm, on CONF-A; EXT is replication).

| Rung | Supported iff |
|---|---|
| 1 (M0) | d PRESENT, and λ1 ABSENT, and IIA(M0) beats the floor |
| 2 (M1) | λ1 PRESENT with positive sign under do(Priority), and IIA(M1) − IIA(M0) ≥ 0.05 |
| 3 (M2) | λ1 PRESENT with positive sign under do(Arousal), the placebo index ABSENT, a positive interaction, and IIA(M2) exceeds both IIA(M0) and IIA(M1) by ≥ 0.05 |

**Evidence against arousal-gated competition in this model** (PDF, decided now): rung 3 not supported and the do(Arousal) slope equivalent to zero (TOST, ±0.05); or the placebo index gates sharpness as strongly as the real one; or the effect appears in the untrained control.

## 8 · The four-outcome table (softmax × sigmoid)

The PDF defines "pattern present" by intervention. The table is therefore **decided in `04`, not in `03`** **[clarification: the handoff says `03`]**. `03` reports the rung-0 verdict per arm in the same layout, labelled "correlational analogue".

- Softmax "pattern" = λ1 PRESENT under do(Priority). Sigmoid "pattern" = c PRESENT (negative) or λ1 PRESENT.
- If either arm is SEED-DEPENDENT, no cell is assigned and the result is reported as unstable across seeds.
- The table is read as below only when the study outcome of 6.2 is "sigmoid ALIVE". Under "attention contributes little in either arm" or "sigmoid NOT ALIVE", no cell is claimed and 6.2 says what is reported instead.

| Softmax | Sigmoid | Reading (PDF) | Check that must pass before the cell is claimed |
|---|---|---|---|
| Present | Present | Learned, normaliser-independent competition | Both arms pass the gate (done); sham and untrained controls clean in both |
| Present | Absent | Competition scaffolded by the zero-sum constraint | Study outcome "sigmoid ALIVE" (6.2), no drift (6.3) |
| Absent | Present | Floor effect in softmax, or a different (gating) solution | Fraction of α < 1e-6 in S; softmax re-analysed on Δz; sigmoid mass regression (6.4) |
| Absent | Absent | No learned selective competition (M0) | Graded: "equivalent to zero" if every analysed head is TOST-equivalent (±0.05) in at least 4 of 5 seeds; "indistinguishable from controls" if not beyond the band but not equivalent; "underpowered" if the CIs are wider than ±0.05 while the band is below 0.05 |

## 9 · Storyline triggers

A storyline is claimed only if its trigger fires on the data named and every check passes. Ranking in `STORY.md` (Phase 4): storylines resting on confirmatory results rank above those resting on exploratory ones.

| # | Headline | Trigger | Checks |
|---|---|---|---|
| S1 | Attention heatmaps in a clinical Transformer fail a sanity check | Softmax rung-0 verdict ABSENT (5.6) | Gate passed; surrogate-severity controls small; narrow or broad wording decided by 5.7; sigmoid verdict reported beside it |
| S1b | The model's prediction barely depends on attention, under either normaliser | Study outcome "attention contributes little in either arm" (6.2) | Gate passed in both arms (done); the uniform-row knockout reported beside it; `kernel_raw` shown as exploratory support only |
| S2 | The normaliser decides | A cell of the four-outcome table is assigned (section 8) | That cell's own check |
| S3 | A network never forced to budget learns one | Study outcome "sigmoid ALIVE" (6.2), and κ's CI inside [0.5, 1.5] in 5 of 5 seeds (7.5) | d PRESENT in the sigmoid arm; 6.4 reported beside it. If only 6.4 holds, the wording is "total mass is insensitive to window composition", not a budget |
| S4 | KA-Transformer protocol ladder | Dropped (A5) | n/a |
| S5 | Heterogeneity | With cluster-robust errors (5.9), in the softmax primary seed: I² ≥ 0.75 in at least half of the heads, or the 95 % prediction interval includes 0 while the pooled CI excludes it in at least half of the heads | Subgroup search on DISCOVERY (septic vs non-septic stay, Unit1/Unit2, density tertile, mean-severity tertile), Benjamini–Hochberg q = 0.10 across heads × subgroups, confirmed on SCREEN (same sign, CI excluding 0). Exploratory wording |
| S5b | Cross-hospital | Performance part is already measured (`01b`). Attention part: the set-B screen (section 12) gives the same verdict as the same proxy on MIMIC | Matched stratum primary (A13); single seed, so "consistent with", never "confirmed" |
| S6 | Diagnose, then fix | A Track D candidate passes the scorecard (section 10) on VAL and on CONF-B | Scorecard |

## 10 · Track D scorecard (`05`)

**Choice of candidates.** By the brief's decision table, applied to the results of `03`/`04`:

| Observed | Candidate |
|---|---|
| Softmax rung-0 ABSENT | Variable-level tokens (STraTS-style); kernel attention second |
| Four-outcome cell Absent/Present, or floor-effect fraction above 5 % | Sigmoid or entmax-1.5 as the explanation layer |
| Rung 1 supported, rungs 2–3 not | No new model: explain with Δz or norm-weighted attention instead of raw α |
| S5 trigger fires | Every explanation reported with seed and bootstrap stability bands |

At most two new architectures. The choice, each candidate's primary explanation method and its hyperparameters are written to `Docs/PREREG_ADDENDUM_TRACK_D.md` and committed **before any `05` training**. SOFA columns are never inputs or targets. Hyperparameters are the matched softmax model's; no tuning.

**Scorecard.** Five seeds per candidate. Item 1 is mandatory; the winner is the candidate with the most of items 2–4, ties broken by the larger comprehensiveness margin.

| # | Criterion | Pass iff |
|---|---|---|
| 1 | Performance | Gate D1 passed, and the lower 95 % bound of AUROC(candidate) − AUROC(softmax, same seed) is above −0.01 at the primary seed (stay-level paired bootstrap, mean of horizons), and the mean difference over seeds is above −0.01 |
| 2 | Sanity (Adebayo-style) | The candidate's primary explanation, run through the section 5 screen against its own untrained and v3-protocol shuffled-label controls, is PRESENT by the seed rule |
| 3 | Faithfulness | Comprehensiveness (drop in the t+6 logit when the top 20 % of hours or tokens by attribution are set to "not measured") exceeds both random-order deletion and raw α of the standard softmax model, each with a subject-level paired-bootstrap CI excluding 0. Sufficiency and deletion/insertion curves (10 % steps) are reported |
| 4 | Stability | Mean pairwise Spearman correlation of the explanation ranking across the 5 seeds, on the same stays, is at least 0.5 and at least that of raw α in the standard model |

Methods compared on every candidate: raw α, clr, norm-weighted, rollout (softmax only), Integrated Gradients, gradient × input. Items 1–3 are recomputed once on CONF-B with everything frozen; a candidate passes only if CONF-B agrees with VAL.

## 11 · Statistical conventions

- Unit of inference: the stay; when a subject has several stays the bootstrap resamples subjects.
- Model comparisons: stay-level paired bootstrap. Training claims: seed-level variance; an effect smaller than the seed SD is "not resolved".
- Effect sizes with 95 % CIs are the headline; p-values are secondary. Nulls are tested by equivalence (TOST against the SESOI). FDR (Benjamini–Hochberg) within each sensitivity family.
- A null is reported as a null at the right scope and is never hidden.

## 12 · Exploratory analyses (labelled as such wherever shown)

- Kernel attention (`kernel`, 3 seeds; `kernel_raw`, 1 seed): the section 5 screen with its v3-protocol shuffled-label control, and total mass by position plus the two knockouts of 6.2 for `kernel_raw` (the probe value 0.039 came from the shortest stays and is re-measured).
- Reversed residualisation (severity residualised on deviation).
- Trained-minus-init contrast per head (possible because the untrained control is the same network at step 0).
- PhysioNet set B attention replication: partial-SOFA proxy from MAP, platelets, bilirubin and creatinine, the same code on both cohorts; reduced-feature models (`r33_softmax_s42`, `r33_sigmoid_s42`) with their own `init.pt` and a v3-protocol shuffled-label control trained on MIMIC FIT; matched stratum primary, all stays beside it (A13); compared with the same proxy on MIMIC VAL, never with full SOFA.
- S2/S3/S5 patch sites on LOC; a one-stage mixed model on 300 stays as a check of the two-stage estimator.
- Optional extras (aux-head forecast vs carry-forward, GRU-D/LSTM baseline, Kamran-style sensitivity): performance analyses on FIT/VAL only; they touch no rule in this file.

## 13 · Items reviewed by Miaki before freezing (decided 2026-11-15)

Each item is a place where this file fills a gap, tightens a PDF default or resolves a conflict.

| # | Item | Section | Decision |
|---|---|---|---|
| 1 | O1 rewording: exact checks E1–E4; the set j < k becomes a query-side reference set; the **sham donor** (routine → routine splice) is the empirical noise floor | 7.1, 7.3 | Approved as written |
| 2 | Seed rule: excess count against the two control networks, PRESENT = 5 of 5 seeds, ABSENT = at least 4 of 5 | 5.6 | Approved as written |
| 3 | "Collapsed": attention knockout must cost ≥ 0.01 AUROC, and mass ≥ 0.05 in at least one layer on analysis queries; arm alive iff ≥ 4 of 5 seeds | 6.2 | Approved with an addition by Miaki: if the paired softmax knockout also costs little, the result is "attention contributes little in either arm", a pre-specified outcome of its own, and 6.2 states how the four-outcome table is then read. Reason: `kernel_raw` had near-zero probe mass and still reached 0.796 |
| 4 | The four-outcome table is decided in `04` by intervention; `03` gives a correlational analogue | 8 | Approved as written |
| 5 | EXT is loaded once, at the very end, and that one pass includes the replication the PDF asks for | 3 | Approved as written |
| 6 | IIA: match rule for a model that predicts a constant vector; margin 0.05 for IIA and DAS | 7.7, 7.8 | Approved as written |
| 7 | Causal effects standardised by the row's logit SD so SESOI 0.05 applies; κ intervals [0.5, 1.5] and [−0.25, 0.25] | 7.4, 7.5 | Approved as written |
| 8 | Pair constraints: t ≥ max(8, k + 6); donor tolerance 1 FIT SD on previous-hour vitals; S1 primary, S0 beside it | 7.1, 7.2 | Approved as written |
| 9 | CONF-A / CONF-B by the sha1 parity rule | 3 | Approved as written |
| 10 | Soup is the primary checkpoint, best single epoch a sensitivity analysis | 3 | Approved as written |
| 11 | One shuffled-label model per arm serves all five seeds (no new control training) | 3, 5.4 | Approved; limitation added in 5.4 (control seed variance is not estimated) |
| 12 | Track D thresholds (20 % deletion, Spearman 0.5, at most two architectures) and the S5 thresholds (I² ≥ 0.75, q = 0.10) | 9, 10 | Approved as written |

## 14 · Traceability: every "→ PREREG" item in `DECISIONS.md`

| Item | Where |
|---|---|
| A2 shuffled-label control and its reading | 3, 5.4, 5.7 |
| A3 carry-forward of the top 6 | 4, 5.8, 7.5 |
| A6 CONF-A / CONF-B | 3 |
| A9 sigmoid fallback | 3 (not triggered) |
| A10 seed rule | 5.6 |
| A11 early hours, Δz as cross-arm quantity, mass by position | 5.1, 6.1, 7.4 |
| A12 heterogeneity with cluster-robust errors | 5.3, 5.9 |
| A13 PhysioNet matched stratum | 9 (S5b), 12 |
| O1 reworded nulls | 7.3 |
| D1 gate | 3 |
| Sigmoid alive / attention contributes little in either arm (PDF Part D check) | 6.2, 8, 9 (S1b) |

## 15 · References used by this file

§10 of the brief, with two substitutions decided by Miaki on 2026-11-15: **R13** is Greenacre, M. (2021), *Compositional data analysis — linear algebra, visualization and interpretation*, arXiv:2110.12439 (log-ratio transforms, clr); **R18** is Geiger, A., et al. (2025), *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability*, JMLR 26, 1–63 (interchange interventions, IIA). Method sources: R14 (random-effects pooling), R15 (TOST, SESOI), R16 (norm-weighted attention), R17 (surrogate data), R19 (DAS), R20 (subspace-patching illusion), R21 (activation patching), R22 (random-network control, linear alignment), R23 (real-pair counterfactuals), R24 and R25 (M2 and M0 framing), R4 (randomisation checks), R5 (sigmoid attention), R29 (comprehensiveness, sufficiency). All are in `Resources/`.

Added by Miaki on 2026-11-15, outside §10's numbering, both in `Resources/` and checked: Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate: a Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society B*, 57(1), 289–300 (FDR control, sections 5.10 and 9). Riley, R. D., Higgins, J. P. T., & Deeks, J. J. (2011). Interpretation of random effects meta-analyses. *BMJ*, 342, d549 (prediction intervals, section 5.9).
