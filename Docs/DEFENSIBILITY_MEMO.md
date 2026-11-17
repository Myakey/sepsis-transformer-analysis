# Defensibility memo: "Does Attention Track Severity?"

Written 2026-11-17 from the committed outputs of notebooks 00–04, `Docs/` and the papers listed in section 9. Companion to `Docs/criticism_matrix.xlsx`; IDs such as A04 or T02 refer to rows there.

**Status of the evidence.** Run and committed: `00`, `01`, `01b`, `02`, `03`, `04` (stages LOC and CONF-A). **PENDING:** `05` (Track D, VAL and CONF-B), `06` (architecture benchmark), `04` stage EXT. Every performance number below is from VAL, the fold used for tuning, early stopping and calibration. Numbers that exist only in `reports/*.json` on Drive were not available to me.

## 1 · Bottom line

The paper that the evidence supports is a **sanity-check paper with a clean negative result and three supporting findings**:

1. In a softmax sepsis Transformer whose predictions demonstrably depend on attention, hour-level attention does not track SOFA severity beyond what untrained and label-shuffled networks show (5 of 5 seeds).
2. Under intervention, attention responds to a spliced severe hour directly and with a common shift, and shows no severity-selective redistribution, in both arms.
3. A matched sigmoid model reaches almost the same AUROC while barely using where it attends.
4. The model does not transport to PhysioNet 2019, and the severity–attention relation is unpredictable at the level of a single stay.

What the evidence does **not** support, and what I recommend cutting or demoting:

| Item | Verdict | Why |
|---|---|---|
| "The normaliser decides" (storyline S2) as a headline | **Cut as a headline.** Keep the facts in one symmetric table | The Absent / Present cell follows from definitions that differ by arm (A08) |
| "Sigmoid attention tracks severity, softmax does not" | **Soften** to a variable-specific statement | On the common variable the sigmoid result is seed-dependent (A05) |
| IIA and the rival models M0/M1/M2 as evidence about mechanism | **Demote** to "inconclusive" in a secondary section | The match rule cannot be satisfied often even when M0 is true (A10) |
| Arousal-biased competition (rung 3) | **Demote** to a secondary section or appendix | Not supported; untrained networks match trained ones (A17) |
| The sign-test p-value (1/32) | **Cut** | Seeds are not independent (T02) |
| KA-Transformer comparison, balancing, architecture benchmark | **Appendix** | Context, not contribution (O02, O03) |
| Any clinical-utility or early-warning language | **Cut** | Not assessed (S09) |

## 2 · The claims, as narrowly as the evidence allows

Residual risk is my estimate of how likely a competent hostile reviewer is to remain unconvinced after the rebuttal: low, medium or high.

### Claim 1 · Performance context

> On MIMIC-IV, under a subject-exclusive chronological split with a causal mask and training-fold-only preprocessing, the softmax model reaches a validation AUROC of 0.806 ± 0.003 over five seeds for sepsis onset within 6 hours, 0.03–0.05 above a LightGBM reference at every horizon. Discrimination is 0.66–0.67 in the first 12 ICU hours and 0.87 after hour 48. The model does not transport: 0.69–0.70 on PhysioNet 2019.

- **Evidence.** `02` Cell 6 (seeds, gate D1); `01` Cells 6–9 (by ICU hour, thresholds); `01b` Cells 8–11 (external); 00 tests 1, 7, 8 (leakage, folds).
- **Strongest argument.** The protocol removes the usual sources of optimism, the result reproduces over five seeds (SD 0.003), three attention variants land at the same level (softmax 0.806, kernel 0.805, KA 0.807), and the paper reports its own weak spots (early hours, external drop, precision 6–9%).
- **Strongest counterargument.** Every number comes from the fold that selected the hyperparameters, the stopping epoch, the checkpoint average and the calibrator (T06). The reference model was not tuned (S11). The label's derivation is undocumented (S01), measurement flags can carry clinician suspicion (S02), and no evaluation is restricted to hours before suspicion of infection (S03), the analysis under which the Epic model fell from 0.62 to 0.47 (Kamran et al., 2024).
- **Rebuttal.** The paper uses performance only as a gate for the interpretability analysis. EXT is reserved for one frozen-model pass. The pre-suspicion evaluation needs no GPU because the cohort already has `HoursSinceSusp`.
- **Residual risk.** **Medium** until EXT and the pre-suspicion evaluation are in; **low** afterwards, provided the text makes no utility claim.

### Claim 2 · The softmax null (the main result)

> In the softmax model, no attention head's association with SOFA severity exceeds the pre-specified control band in any of five seeds (0 of 32 heads each; excess E = −8, −7, −8, −8, −8). The same verdict holds with norm-weighted attention, with the best single epoch, and when the budget-matched or the auxiliary-task-free shuffled-label control replaces the primary one.

- **Evidence.** `03` Cells 5, 6, 11, 12; PREREGISTRATION §5.
- **Strongest argument.** This is the randomisation logic of Adebayo et al. (2018) applied to a clinical time-series model with the thresholds, controls, stays and seed rule fixed in advance. The result is identical across seeds, checkpoints and three dependent variables. The trained heads do carry real associations (largest |b1| ≈ 0.25), so the finding is not "nothing there"; it is that networks which never learned sepsis show as much.
- **Strongest counterargument.** The band (0.346) is set by **one** label-shuffled network, trained for 4 epochs on 30% of the training fold, which itself has 7–8 passing heads. Its seed variance is unknown (A04). Against the budget-matched control the count is 0 versus 0. The seed rule's p-value is invalid (T02). Only 39 of the 800 discovery stays are septic, and an exploratory comparison finds different slopes in septic stays for 21 heads (A15). There is no positive control showing the screen can detect a planted effect (A12).
- **Rebuttal.** The dependence on one control was stated as a limitation before any v4 result (PREREGISTRATION §5.4). The pre-specified reading is the narrow one: "sepsis labels add nothing detectable beyond physiology learning", and the broader reading was allowed only if the auxiliary-task-free control agreed, which it did. Three cheap analyses close most of what remains: more control seeds, a signal-injection check, and a septic-enriched screen.
- **Residual risk.** **Medium** now; **low** if the control-seed and signal-injection analyses come back as expected. If the band moves a lot across control seeds, the claim has to be restated as "not distinguishable from a label-shuffled network that learned to forecast physiology", which is still publishable.

### Claim 3 · Attention is used

> The softmax model's predictions depend on its attention layers and on their pattern: zeroing the attention output costs 0.13–0.16 AUROC, and replacing each row with a uniform row costs 0.16–0.18, in every seed.

- **Evidence.** `03` Cell 9.
- **Strongest argument.** It removes the easy dismissal of Claim 2 ("attention is ignored, so of course it tracks nothing"). The pre-registration had a separate outcome for that case, and it did not occur.
- **Strongest counterargument.** Zeroing a sublayer takes the network off its training distribution, so the drop overstates dependence.
- **Rebuttal.** The pre-registration concedes this (§6.2). The uniform-row knockout keeps the sublayer active with the same total mass and costs even more, so the pattern itself matters.
- **Residual risk.** **Low.**

### Claim 4 · What interventions show

> Replacing one past hour with a matched severe hour from another patient, relative to a sham replacement, changes the logit to that hour in half or more of the 24 heads of layers 2–4 in both arms (excess over controls: softmax E = 15–20, sigmoid 8–18), lowers the logits of the other hours in a subset of heads (softmax 8–13, sigmoid 2–5), and produces no severity-selective redistribution in any head of any of the ten models. For the selectivity slope the sigmoid arm is equivalent to zero (±0.05) and the softmax arm is underpowered. The predicted risk moves little (−0.08 SD at the primary softmax seed).

- **Evidence.** `04` Cells 7, 13, 17 (LOC → CONF-A, both arms, five seeds); exact checks E1–E4 all 0.0.
- **Strongest argument.** Thresholds were fixed on LOC and confirmed once on CONF-A; every effect is a severe-minus-sham contrast on the same stay and hours; four exact identities hold to 0.0 in every model, which rules out implementation leaks.
- **Strongest counterargument.** A spliced hour from another patient is not a physiological trajectory (A09). The output hardly moves and moves downward, so the manipulation may be irrelevant to how the model predicts (A14). The softmax null on selectivity is not an equivalence result (T08). The pre-registered sites S4 and S6 are missing from the committed output (A11).
- **Rebuttal.** The sham splice carries the same disturbance without a change in severity class and is subtracted. Loosely matched and tightly matched donors agree across heads (r = +0.979). The small output effect is reported as a result: the intervention tests how attention is allocated, and the paper should say so in those words.
- **Residual risk.** **Medium.** The descriptive statement is safe. Any mechanistic reading beyond "direct response, common shift, no selectivity" is not.

### Claim 5 · The sigmoid arm

> A sigmoid arm matched on initial weights, data order and hyperparameters scores 0.007 AUROC lower (5 of 5 paired seeds). Its total attention mass settles at 0.36–0.68 per layer. Its gates are independent (compensation κ between −0.15 and +0.05, every interval inside ±0.25); it did not learn a budget. Replacing its attention rows with uniform rows of the same mass costs 0.01–0.02 AUROC in four of five seeds (0.07 in the fifth). On its pre-specified screening variable (log a) its severity association exceeds controls in 5 of 5 seeds; on the variable shared with the softmax arm it is seed-dependent.

- **Evidence.** `02` Cells 6, 8; `03` Cells 6, 8, 9; `04` Cell 13.
- **Strongest argument.** The pair is matched as tightly as two arms can be (step-0 parameter equality is a test in `00`; a warmup control rules out the one remaining difference). The findings "independent gates" and "pattern barely used" are clear in the data and are not what the methodology hoped to find, which speaks for the honesty of the reporting.
- **Strongest counterargument.** Ramapuram et al. (2025) train sigmoid attention with LayerScale unless stated otherwise; this arm has none, so it may be a poorly trained instance of sigmoid attention (A07). The "alive" criterion was passed on a knockout that the uniform-row result undercuts. The PRESENT verdict on log a may reflect total mass, which does depend on the window's severity composition (A05).
- **Rebuttal.** The matching rule forbade arm-specific stabilisers, the arm passed the same performance gate, and the LayerScale fallback was pre-registered and not triggered. The paper reports the uniform-row numbers next to every pattern-level statement and makes no claim that the normalisers differ in severity tracking.
- **Residual risk.** **Medium to high** for any sentence that reads the sigmoid arm as a mechanism contrast; **low** for the descriptive claim above.

### Claim 6 · Heterogeneity

> The severity–attention relation varies widely between stays (median I² 0.96 with cluster-robust errors). For 88% of softmax heads whose pooled interval excludes zero, the 95% prediction interval for a new stay includes zero.

- **Evidence.** `03` Cell 10. Method: prediction intervals as in Riley et al. (2011).
- **Strongest argument.** It is the patient-level version of Claim 2: even where an average association exists, its sign in the next patient is unpredictable, which is what matters for anyone shown a heatmap for one patient.
- **Strongest counterargument.** One seed. Stage-1 regressions treat hour pairs as independent, which inflates I²; the subgroup search that follows is exploratory with 39 septic stays.
- **Rebuttal.** The cluster-robust estimate is the one quoted (0.957 against 0.994 unadjusted). The subgroup results are labelled exploratory and used only to motivate the septic-enriched screen.
- **Residual risk.** **Low**, as a secondary finding marked single-seed.

### Claim 7 · Rungs 1–3 and the high-level models

> No pre-specified high-level model (fixed sharpness, saliency-dependent sharpness, arousal-dependent sharpness) beat its floor by the pre-specified margin in either arm, and intervening on a heart-rate-plus-respiratory-rate index did not sharpen attention beyond controls by the seed rule (softmax absent, sigmoid seed-dependent; untrained networks show as many passing heads as trained ones). We treat the interchange analysis as inconclusive.

- **Evidence.** `04` Cells 13–14; DEVIATIONS P1, P2; DECISIONS A25, A28.
- **Strongest argument.** The outcome was reported exactly as pre-specified, including a deviation that was logged before the confirmatory fold was opened and a second floor reported beside the one in force.
- **Strongest counterargument.** The interchange analysis is the weakest part of the study. The match rule for a model that predicts "no structure" was invented here, and it is met in only 35–40% of sham splices; for five untied keys, chance alone gives 52%. A model that is exactly right could not score well. Sutter et al. (2025) show that interchange accuracy can be high for randomly initialised networks under flexible alignments, and no random-network value is reported here. Makelov et al. (2023) describe how subspace patching can mislead; the subspace search (DAS) did not run.
- **Rebuttal.** None that rescues it as evidence. The honest position is that the selectivity slope carries the inference and the interchange numbers are reported for completeness.
- **Residual risk.** **High** if presented as a result about mechanism; **low** if presented as inconclusive in a secondary section.

### Claim 8 · Track D (PENDING)

Conditional wording, by outcome of the pre-specified scorecard (`05`, VAL then CONF-B):

| Outcome | What may be claimed |
|---|---|
| A candidate passes items 1–3 on VAL and on CONF-B | "A readout whose contributions are exact by construction passed the randomisation and deletion checks that raw attention failed, at an AUROC cost below 0.01." Nothing about clinical usefulness |
| It fails item 1 (performance) | "The exact readout cost more than 0.01 AUROC." |
| It passes item 1 and fails item 2 or 3 | "Exactness of the decomposition did not make the explanation distinguishable from controls." This strengthens Claim 2 |
| VAL and CONF-B disagree | Report both; no claim |

In every case the trunk remains opaque (X02), and the phrase "interpretable by design" should not appear without that qualification.

### Claim 9 · Method

> We provide a test protocol for attention claims in clinical time-series models: a frozen analysis plan, control networks sharing initial weights, surrogate-severity controls, sham interventions, exact identities that must hold to floating-point precision, and fold access enforced in code.

- **Strongest argument.** It is concrete, reusable, and the code enforces it (`data.FoldGuard`; 00 test 7).
- **Strongest counterargument.** The plan was not lodged with an external registry, parts of it were written after earlier results were known, and plan and code were drafted with an AI assistant (T03, O01).
- **Rebuttal.** The plan states what was known when it was written (§1), deviations are dated against the data, and the repository history can be archived. Disclosure of AI assistance is a paragraph.
- **Residual risk.** **Medium** until the archive, the deviations table and the disclosure exist; **low** afterwards.

## 3 · What the paper must not claim

1. **Clinical utility, early warning, or prediction before clinicians suspect sepsis.** Precision is 6–9% at the F1 threshold; no alert-burden, lead-time or pre-suspicion analysis exists (S03, S09).
2. **State of the art, or superiority to the KA-Transformer paper.** Their figures are for one prediction per patient with length of stay as an input. Under this protocol the two architectures are indistinguishable (−0.001 [−0.008, +0.006]).
3. **"Attention is not explanation" as a general statement, or "attention is meaningless".** The claim is about SOFA severity, in this model, on this cohort. Attention matters for the prediction (Claim 3).
4. **"Trained attention is the same as untrained."** Trained and untrained slopes differ in 24 of 32 heads (exploratory). The claim is "not beyond the control band".
5. **"No association" or "zero effect" at rung 0.** Only 3–9 of 32 heads are equivalent to zero.
6. **"The normaliser decides", "the sigmoid network found a different mechanism", or "softmax competition is an artefact".** On logits both arms behave alike (A08).
7. **"A network never forced to budget learns one."** The opposite was found: independent gates, κ ≈ 0.
8. **That allocation by normalisation (M0) was confirmed or refuted.** The interchange test could not tell (A10).
9. **Anything about arousal-biased competition as a theory**, or the exploratory interaction (softmax direct effect larger in naturally aroused hours) as evidence. It is observational in arousal and confounded with acuity.
10. **Generalisation of the attention findings to other hospitals.** The set-B screen is exploratory, one seed, proxy severity: "consistent with".
11. **That the findings describe septic patients or the first hours of a stay.** The analysis uses stays of at least 24 hours and queries from hour 8; half of septic stays have their onset before hour 4.
12. **"Pre-registered" without qualification.** Say "pre-specified and frozen in a version-controlled repository; not registered externally".
13. **A p-value for the 5-of-5 seed rule.**
14. **Named heads** (for example "L4H3 encodes severity"). Head identity is not stable across seeds.
15. **That one normaliser predicts better.** −0.007 is consistent and negligible.
16. **"Interpretable by design" for Track D** without saying that only the readout is additive.
17. **"First"**, until the related-work search is done (O04).
18. **Any EXT or CONF-B number** before those passes are run once and committed.

## 4 · Are methods from NLP interpretability justified here?

**Where they are.**

- **Randomisation controls.** Comparing a statistic on the trained network with the same statistic on a network with random weights or shuffled labels (Adebayo et al., 2018) assumes nothing about the input modality. It is the backbone of Claim 2.
- **Normaliser algebra.** The centred log-ratio removes the softmax denominator exactly for any softmax row, and weighting attention by the norm of the transformed value vector (Kobayashi et al., 2020) is a property of the attention sublayer, not of words.
- **Interventions on a deterministic network.** Splicing an input or an embedding and reading the change in logits needs only a computational graph. Here four exact identities (repeat-run, causality, layer-1 locality, query-side isolation) hold to 0.0 in every model, which is a stronger implementation check than most NLP studies report.
- **The debate's core logic.** That weights need not track importance (Jain & Wallace, 2019; Serrano & Smith, 2019) is a statement about the architecture. Jain and Wallace's own tasks include two built from MIMIC discharge summaries, so the clinical domain was in the debate from the start, though as text.

**Where they are not, or only weakly.**

- **The token.** A word is a nameable unit. An ICU hour is a vector of 36 variables with their missingness flags and gaps. Attention over hours cannot say which variable mattered, so even a faithful hour-level map answers a coarser question than a clinician asks. Ismail et al. (2020) found on synthetic time series that saliency methods often fail to identify which feature matters at which time step, across recurrent, convolutional and Transformer models. That literature, not the NLP one, is the closer precedent, and the paper currently cites almost none of it (O04).
- **Counterfactual pairs.** In text, minimal pairs are natural. A spliced hour from another patient is a discontinuity in a physiological trajectory. Matching and the sham control reduce the problem; they do not remove it. Heimersheim and Nanda (2024) stress that what a patching experiment shows depends on its design, starting with the direction of patching (noising or denoising).
- **Interchange accuracy and causal abstraction.** The framework (Geiger et al., 2025) presumes a high-level algorithm with well-defined variables. Here the high-level models come from an analogy with cognitive psychology, the variables are continuous, and the match rule had to be invented. This is the one place where the transfer fails outright (Claim 7).
- **Faithfulness versus plausibility.** Jacovi and Goldberg (2020) ask that faithfulness be defined and evaluated separately from plausibility. "Does attention track severity" is a plausibility question. The paper must say so and reserve "faithfulness" for the deletion tests of Track D and the knockouts.
- **Deletion tests.** Setting an hour to "not measured" is closer to natural data here than deleting a word is in text, because unmeasured hours occur. But missingness is informative in ICU data, so deletion also injects a signal. Track D's comprehensiveness scores (definitions from DeYoung et al., 2020) inherit that ambiguity.
- **No positive control yet.** Nothing shows that the screen detects a severity-tracking head of known size in this setting. The signal-injection check (A12) is the cheapest way to answer the transfer question empirically.

## 5 · Why a model with AUROC ≈ 0.80 is a valid object of study

1. **The question is about the model, not about sepsis.** Whether a heatmap reflects a property of the network is well posed for any network that has learned real signal. The gate establishes that: +0.03 to +0.05 AUROC over LightGBM with every lower bound above zero, AUPRC 5.7–5.9 times the base rate, and label-shuffled controls at 0.47–0.53.
2. **The component under study is in use.** Removing attention costs 0.13–0.16 AUROC. The objection "a weak model's attention is noise" would predict a small knockout cost.
3. **This is the performance range of models that reach clinicians.** The Epic Sepsis Model had an AUROC of 0.63 at an external centre (Wong et al., 2021). Fleuren et al. (2020) report 0.68–0.99 for ICU models on retrospective data, with heterogeneous definitions that prevented pooling. A heatmap from a 0.80 model is a realistic thing to be shown at the bedside, so its meaning is worth testing.
4. **Higher published numbers often belong to a different task.** The architecture modelled on Zhu et al. (2025), whose paper reports 0.94–0.98, scores 0.807 under this protocol. Their figures concern one prediction per patient with length of stay as an input. This does not show their numbers are inflated; it shows that a higher number would not, by itself, be a better object.
5. **Where the analysis looks, the model is better than its pooled number.** Attention is analysed at queries from hour 8 in stays of at least 24 hours. By-hour AUROC (t+6) is 0.67 in hours 7–12, 0.73 in hours 13–24, 0.77 in hours 25–48 and 0.87 in hours 49–168. Reporting AUROC on the analysis sample itself is a cheap addition.

**What to concede.** A stronger model might behave differently. At this performance the network may lean on measurement patterns more than on physiology, in which case severity tracking was never to be expected. The scope line ("this model, on this cohort") is the answer, and it has to appear in the abstract.

## 6 · Draft Limitations section

> **Limitations.** *Label and cohort.* Sepsis onset is an operational Sepsis-3 label that depends on clinician actions and on SOFA, which is computed from variables the model receives. Covariation between attention and severity is therefore expected in any network; this is why we test against control networks, and it also means our results are conditional on one implementation of the label. All confirmatory analyses use one centre.
>
> *Performance.* Validation figures come from the fold used for tuning, early stopping, checkpoint averaging and calibration; Table X reports the single evaluation on an untouched fold. Discrimination was 0.66–0.67 in the first 12 ICU hours, and at a threshold chosen for F1 the model found 10–21% of positive hours at 6–9% precision. The model lost about 0.09 AUROC on PhysioNet 2019 and was miscalibrated there. We did not evaluate predictions made only before clinical suspicion, alert burden, lead time or net benefit, and we make no claim of clinical utility. The model receives measurement indicators, which reflect clinician behaviour.
>
> *Scope of the attention analysis.* The correlational analysis covers stays of at least 24 hours and query hours from 8 onward, and the interventions require a readout at least six hours after the spliced hour. Half of septic stays have their onset before ICU hour 4, and 39 of the 800 discovery stays were septic, so our findings mostly describe attention in patients who did not meet sepsis criteria. Severity is SOFA as recorded; other notions of clinical importance were not tested.
>
> *Controls.* The control band was set mainly by one label-shuffled network per arm, shared by all five seeds; [across k further control seeds the band ranged from X to Y]. "Absent" means not distinguishable from controls. It is an equivalence result only where stated.
>
> *Interventions.* Splicing an hour from another patient produces inputs that do not occur naturally; a sham splice controls for the disturbance but cannot make the counterfactual physiological. The splice changed predicted risk little, so the causal results concern how attention is allocated, not how the prediction is formed. The interchange analysis was inconclusive: its match criterion for models that predict no structure is noise-limited.
>
> *Sigmoid arm.* The sigmoid model was trained without the stabilisers recommended for sigmoid attention, to keep the arms matched, and it made little use of its attention pattern. Its dependent variable at the screening stage differs from the softmax arm's, and on the shared variable its result was seed-dependent.
>
> *Analysis plan.* The plan was frozen in a version-controlled repository before the confirmatory folds were opened and was not registered externally. Some thresholds were inherited from an earlier screen of the same architecture. Three deviations are listed in Table S1. Outcomes were not ranked in advance and are not adjusted for multiplicity across outcomes. Several secondary analyses use one seed and are marked.
>
> *Process.* Code and the analysis plan were drafted with an AI coding assistant under the authors' direction; the authors ran every experiment and verified the code with a test suite.

## 7 · Draft "Threats to validity" paragraph

> **Threats to validity.** *Construct validity:* severity is operationalised as SOFA, which shares ingredients with both the inputs and the label, and "attention" is analysed per head at the level of hours, which mix 36 variables; a null on this construct does not exclude attention to something else. *Internal validity:* fold access was enforced in code and four exact identities held to floating-point precision in every model, but the control band rests largely on one control network per arm, and interventions use spliced hours that are matched, not natural. *Statistical conclusion validity:* decisions follow pre-specified rules over five seeds that share data and controls, so the seed rule is a decision rule and carries no p-value; the softmax null on selectivity is underpowered by our own grading; outcomes are not adjusted for multiplicity. *External validity:* one centre, one label implementation, one architecture family and one training recipe; performance did not transport to PhysioNet 2019, and the attention findings were replicated there only in an exploratory, single-seed analysis with a proxy for severity.

## 8 · What would change this memo

| If this happens | Then |
|---|---|
| EXT AUROC is inside the VAL interval | Claim 1 residual risk drops to low |
| EXT AUROC is clearly lower | Quote EXT as the headline number; nothing else changes |
| Pre-suspicion AUROC falls toward 0.5 | Say so prominently; it strengthens the "do not claim utility" position and does not affect Claims 2–7 |
| The band varies widely across control seeds | Restate Claim 2 in its narrow form only |
| Signal injection shows the screen needs an effect far above 0.35 to fire | Claim 2 becomes "underpowered against this band"; report the detectable size |
| The septic-enriched screen is PRESENT | Claim 2 must be scoped to non-septic stays, and the septic result becomes a finding to follow up |
| S4 shows the output does not depend on allocation | Rungs 1–3 become statements about logits only, in the pre-registered wording of §6.2 |
| A Track D candidate passes on VAL and CONF-B | Add Claim 8, first row |

## 9 · References used in this memo

Each was read for this review to the extent stated. Full details and the list of papers I could not access are on the "Sources" sheet of `criticism_matrix.xlsx`.

- Adebayo et al. (2018). Sanity Checks for Saliency Maps. NeurIPS. *PDF in Resources; abstract.*
- DeYoung et al. (2020). ERASER. ACL. *PDF in Resources; metric definitions.*
- Fleuren et al. (2020). Machine learning for the prediction of sepsis: a systematic review and meta-analysis. Intensive Care Medicine. *PDF in Resources; abstract.*
- Geiger et al. (2025). Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability. JMLR 26. *PDF in Resources; abstract and selected passages.*
- Heimersheim & Nanda (2024). How to use and interpret activation patching. arXiv:2404.15255. *PDF in Resources; §2.3.*
- Ismail, Gunady, Corrada Bravo & Feizi (2020). Benchmarking Deep Learning Interpretability in Time Series Predictions. NeurIPS. *Read on the web.*
- Jacovi & Goldberg (2020). Towards Faithfully Interpretable NLP Systems. ACL. *Abstract and opening sections, on the web.*
- Jain & Wallace (2019). Attention is not Explanation. NAACL-HLT. *PDF in Resources; abstract, datasets.*
- Kamran et al. (2024). Evaluation of Sepsis Prediction Models before Onset of Treatment. NEJM AI 1(3). *PDF in Resources; abstract, introduction, methods.*
- Kobayashi et al. (2020). Attention is Not Only a Weight. EMNLP. *PDF in Resources; abstract.*
- Makelov, Lange & Nanda (2023). Is This the Subspace You Are Looking for? arXiv:2311.17030. *PDF in Resources; abstract.*
- Ramapuram et al. (2025). Theory, Analysis, and Best Practices for Sigmoid Self-Attention. ICLR. *PDF in Resources; abstract and passages on the bias term and LayerScale.*
- Riley, Higgins & Deeks (2011). Interpretation of random effects meta-analyses. BMJ 342:d549. *PDF in Resources; summary points.*
- Serrano & Smith (2019). Is Attention Interpretable? ACL. *Read on the web.*
- Sutter, Minder, Hofmann & Pimentel (2025). The Non-Linear Representation Dilemma. arXiv:2507.08802. *PDF in Resources; abstract.*
- Wong et al. (2021). External Validation of a Widely Implemented Proprietary Sepsis Prediction Model. JAMA Internal Medicine 181(8). *PDF in Resources; abstract, methods.*
- Zhu et al. (2025). Application of the KA-Transformer model to early sepsis prediction. Discover Applied Sciences. *PDF in Resources; abstract.*
