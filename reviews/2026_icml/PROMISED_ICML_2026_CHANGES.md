# PROMISED_ICML_2026_CHANGES.md

Changes promised in the ICML 2026 rebuttals that must be implemented in the revised manuscript.

---

## Introduction

- [ ] Rewrite narrative arc: prior concerns → we discovered a *new* problem (parameter ambiguity) → conclusions survive → we explain why → general framework
- [ ] Linger on the surprise before resolving it (currently moves too fast from "ambiguous" to "robust")
- [ ] Clearly distinguish our contribution from prior concerns (wide CIs, approach-3 inconsistencies, Kaplan-Chinchilla discrepancy)

## Abstract

- [ ] Replace "renewed confidence in Chinchilla as a durable guide for scaling modern models" with language scoped to the Chinchilla dataset and fitting methods

## Main Text — Analytical Results (from Appendix C)

- [ ] Summarize key analytical results from Appendix C in the main text:
  - Multiplicative errors: $\hat{A} \approx A c_m^\alpha$, exponent invariant
  - Additive errors: variable effective slope $N/(N+c_a)$, fitter compensates by shifting $\hat{\alpha}$
  - Systematic bias: $\hat{\alpha} = \alpha/s$ with $R^2 > 0.999$

## Discussion — New Sections/Paragraphs

- [ ] **Diagnostic Framework paragraph**: decision tree for diagnosing D/N ratio behavior:
  - D/N shifts up/down but stays flat → multiplicative error
  - D/N trends upward with C → positive additive error (e.g., embeddings included)
  - D/N trends downward with C → negative additive error or systematic bias ($s > 1$)
  - D/N becomes noisy but median holds → random measurement error

- [ ] **Formula Recommendation**: recommend Standard Formula with three reasons:
  1. Derived from architectural first principles with no free parameters
  2. Consistent with Kaplan et al.'s convention for $C \approx 6ND$
  3. Produces the flattest D/N ratio (slope = -0.572 per decade vs -1.049, -1.248)
  - State the concrete D/N ratio under each formula

- [ ] **Limitations section** (or expand Scope paragraph):
  1. Analysis conditioned on Chinchilla dataset and fitting methodology
  2. Only parameter-count perturbations, not data-side or optimizer-side errors
  3. Synthetic perturbations, not ground-truth alternative measurements
  4. Parameter counts chosen because they admit clean counterfactual analysis; data quality lacks a canonical ground-truth metric

- [ ] **Code validation statement**: explicitly state we reproduced Epoch AI's published fit parameters; note reliance on their implementation as a limitation alongside the validation steps taken

- [ ] **Implications for scaling laws** paragraph: power-law scaling appears to capture regularities robust to moderate covariate misspecification (at least within the Chinchilla regime)

## Section 3 — Robustness Analysis

- [ ] Revise overclaiming sentence "all four sensitivity analyses demonstrate that Chinchilla's key results withstand sizable perturbations" → "...withstand perturbations beyond the 15.2% discrepancy we identified, with the nature and threshold of breakdown depending on the perturbation type (see Sections 3.1–3.4)"
- [ ] Explicitly discuss the threshold at which each perturbation type begins to qualitatively change results

## Figure 5

- [ ] Add shaded vertical band or annotation indicating the realistic perturbation range (~15%) so readers can immediately see the realistic regime is within the robust zone

## Future Directions

- [ ] Reframe from "we could do this for other scaling laws" to a concrete methodological call to action: every scaling law paper deriving prescriptions — Muennighoff et al. (data repetition), Gadre et al. (overtraining), Sardana et al. (inference-efficient), MoE scaling, etc. — should assess sensitivity of their prescriptions to covariate specification errors
- [ ] Note extension to data-side covariates (token counts, data quality metrics) as a concrete future direction

## Reporting Checklist (new)

- [ ] Add recommendation that scaling law papers should:
  1. State the exact parameter-counting formula used
  2. Specify whether embedding parameters are included in N
  3. Specify whether tied weights are counted once or twice
  4. Report both the $C \approx 6ND$ approximation and the exact FLOP count

## Related Work / Differentiation

- [ ] Sharpen differentiation from Porian et al. and Pearce & Song: we discover an *internal* ambiguity within Chinchilla (not a between-paper discrepancy), build a systematic sensitivity analysis, and derive analytical explanations; our additive-constant analysis is quantitatively consistent with their reported shifts
