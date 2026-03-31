# Rebuttal to Reviewer EBqg

We thank Reviewer EBqg for their careful reading and for acknowledging that the paper is "generally well written and well motivated" and that investigating scaling law robustness is "meaningful and potentially impactful." We appreciate the constructive questions, which have helped us identify places where the manuscript undersells its contributions. We address each concern below.

---

## Practitioner Adoption Does Not Address Our Question

Reviewer EBqg asks whether practitioner adoption constitutes sufficient validation. We argue it does not, because adoption tests a different hypothesis (*does the law work?*) than our paper (*are the law's inputs trustworthy?*). These are orthogonal questions. The fact that LLaMA teams used Chinchilla's prescriptions successfully does not tell us whether those prescriptions would have changed if the parameter counts had been defined differently.

Every single model parameter count in Chinchilla's Table A9 — the dataset on which the field's compute-optimal scaling prescriptions rest — was ambiguous, with discrepancies reaching 15.2%. This went unnoticed for four years. The natural expectation is that a 15% systematic error in a key covariate of a power-law fit should distort the estimated exponents and shift the resulting prescriptions. It does not. Understanding *why* it does not is the intellectual contribution of this paper — and to our knowledge, no prior work has provided a systematic framework for answering: "How wrong can a scaling law's inputs be before its outputs break?"

---

## Correcting the "Purely Empirical" Characterization

The paper does contain theoretical analysis (Appendix B) deriving exact functional dependencies of the fitted scaling parameters on each perturbation type. These are mathematical derivations, not purely empirical observations. We acknowledge that by placing these results in the appendix, we made them too easy to miss. In the revision, we will promote the key analytical results into the main text. We summarize them here:

1. **Multiplicative errors are absorbed into prefactors.** We prove that $\hat{A} \approx A \cdot c_m^{\alpha}$ while the exponent $\hat{\alpha} \approx \alpha$ remains invariant. The D/N scaling trend is therefore mathematically preserved. This is a proof, not an observation.

2. **Additive errors create a variable effective slope.** The effective log-log slope becomes $N/(N + c_a)$, which is no longer constant. The fitter must select a single exponent to represent this varying slope, causing $\hat{\alpha}$ to shift upward when $c_a > 0$ and downward when $c_a < 0$. This analytically explains the quantitative findings of both Pearce & Song (2024) and Porian et al. (2024) as special cases.

3. **Systematic bias rescales the exponent as $\hat{\alpha} = \alpha / s$**, with an empirical fit of $R^2 > 0.999$. This makes the exponent in the compute-optimal ratio $(\alpha/s - \beta)/(\alpha/s + \beta)$, breaking the $\alpha \approx \beta$ condition unless $s = 1$.

These results yield a diagnostic framework with concrete practical implications (discussed further below), moving the paper well beyond a purely empirical study.

---

## A Diagnostic Framework for Practitioners

The reviewer asks whether the paper provides practical guidance for LLM development. It does, and we will make this more explicit in the revision.

Our perturbation analysis yields a diagnostic table for anomalous D/N ratio behavior:

| Observed symptom | Diagnosis | Fix |
|---|---|---|
| D/N shifts vertically but stays flat | Multiplicative counting error (exponent unaffected) | Correct the scaling factor |
| D/N tilts (slopes with compute) | Additive error (e.g., embeddings) or systematic bias ($\hat{\alpha}$ distorted) | Investigate size-independent parameter blocks |
| D/N becomes noisy but median holds | Random measurement error | Increase bootstrap samples |

A practitioner observing an unexpected D/N trend can use this taxonomy to diagnose whether the issue lies in the functional form or the covariate specification.

---

## Which Parameter-Counting Formula Should Be Preferred

The reviewer asks directly which formula should be preferred. We recommend the **Standard Formula** (Eqn. 1) for three reasons, presented in order of importance:

1. **Derived from architectural first principles with no free parameters.** The Standard Formula computes parameters directly from the architectural hyperparameters (layers, dimensions, heads) using the well-established transformer parameter counting convention. It requires no fitting or ad hoc choices.

2. **Consistent with Kaplan et al.'s convention for computing $C \approx 6ND$.** The convention of excluding embedding parameters from $N$ when estimating compute aligns with the Standard Formula, ensuring internal consistency between the parameter count and the compute estimate.

3. **As confirmatory evidence, it produces the flattest D/N ratio with compute.** The slope is $-0.572$ per decade of compute, compared to $-1.049$ (best fit formula) and $-1.248$ (reported parameters). A flatter slope means the ~20:1 heuristic holds more stably across compute budgets, strengthening the Chinchilla finding.

To be clear: while all three formulas preserve the qualitative Chinchilla conclusion (this is the robustness claim), the Standard Formula provides the strongest quantitative stability. We will add this recommendation to the Discussion section in the revision.

---

## The ~20:1 Tokens-per-Parameter Ratio Is Preserved

The reviewer asks whether our findings provide new guidance on the commonly cited 20-tokens-per-parameter rule. They do: under all three counting formulas, the ~20:1 ratio is preserved. What changes is how *stable* that ratio is across compute budgets. As shown above, the Standard Formula yields the most stable ratio, followed by the best fit formula and then the reported parameters. This means that the Standard Formula not only preserves the 20:1 heuristic but makes it more reliable as practitioners scale to larger compute budgets.

This is new guidance: practitioners should use the Standard Formula parameter counts when applying the Chinchilla heuristic, as it produces the most compute-invariant prescription.

---

## Why We Do Not Propose a Modified Scaling Law

The reviewer asks whether our findings suggest a new or modified scaling law. A modified law is not warranted by our findings, because the law is correct as stated; it is the parameter counts that were ambiguous. Modifying the functional form would be addressing the wrong problem. Our contribution is to show that the existing law is more robust than one might fear, and to explain *why* via the mathematical structure of power-law fits.

More constructively, our diagnostic framework (above) reveals when an *apparent* need to modify the scaling law is actually an artifact of additive specification errors in the covariates. This is precisely the guidance that helps the community avoid premature modifications to the law.

---

## Methodological Contribution and Implications

We propose that sensitivity analysis of covariate specification should become standard practice in scaling law research. To our knowledge, no prior scaling law paper has performed systematic covariate sensitivity analysis. Our four perturbation types (multiplicative, additive, systematic bias, noise) map onto universal sources of covariate error that arise in any power-law fit, not just Chinchilla. Many recent papers derive prescriptions from power-law fits — Muennighoff et al. (2023) on data repetition, Gadre et al. (2024) on overtraining, Sardana et al. (2024) on inference-efficient scaling, and MoE scaling laws — each using covariate definitions that could contain analogous ambiguities. Our framework provides the template for stress-testing those prescriptions, and the analytical toolkit (Appendix B) applies directly.

More broadly, our results suggest that power-law scaling captures regularities robust to moderate covariate misspecification. If a foundational result's covariates can be this wrong without affecting its conclusions, the law is fitting something more fundamental than the precise variable we plug in for $N$. We will make this point more prominent in the revision.

---

## Summary of Planned Revisions

In response to Reviewer EBqg's feedback, we plan the following changes:

1. **Promote key analytical results from Appendix B into the main text**, including the proofs for multiplicative invariance, the additive-error effective slope, and the systematic-bias exponent rescaling. This directly addresses the "purely empirical" characterization.

2. **Add a Diagnostic Framework paragraph to the Discussion**, synthesizing the perturbation results into a practical decision tree for diagnosing anomalous D/N behavior.

3. **State a clear recommendation for the Standard Formula** in the Discussion, with the three reasons enumerated above.

4. **Add a paragraph on the meta-scientific insight and methodological call to action**: scaling laws are more robust to covariate misspecification than the field assumes, and every scaling law paper deriving prescriptions should perform covariate sensitivity analysis.

We hope these revisions address the reviewer's concerns.
