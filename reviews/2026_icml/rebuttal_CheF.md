# Rebuttal to Reviewer CheF

We thank Reviewer CheF for their thorough evaluation and substantive questions. We are glad the reviewer finds that the paper provides "a detailed discussion of the Chinchilla scaling laws" and that the perturbation experiments "verify the stability" of the scaling laws. Below, we address each concern in turn.

## Clarifying the Paper's Contribution

We do not propose adjusted or revised scaling laws; rather, we investigate how robust the existing law is to corrected and perturbed parameter counts. Reviewer CheF's summary characterizes the paper as trying to "adjust the Chinchilla scaling laws." This distinction bears on both the novelty and scope of the work.

The core finding is this: every single one of the 50 model parameter counts in Chinchilla's Table A9 — the dataset on which the field's compute-optimal scaling prescriptions rest — was ambiguous, with discrepancies reaching 15.2%. This went unnoticed for four years. The three parameter-counting formulas we evaluate are not alternative scaling laws; they are three equally valid interpretations of the same model architectures. The perturbation analysis then generalizes beyond any specific formula to characterize the sensitivity of power-law fits to covariate misspecification, yielding analytical results that apply broadly. The natural expectation is that getting all 50 input covariates wrong should matter. It does not. Understanding *why* it does not is the intellectual contribution of this paper.

## Novel Contributions

In response to the reviewer's concern about originality, we enumerate the paper's novel contributions:

- **(a) First identification of internal parameter-count ambiguity within Chinchilla's own Table A9.** Prior works (Porian et al., 2024; Pearce & Song, 2024) identified specific accounting decisions explaining discrepancies *between* Chinchilla and Kaplan. We discovered a previously unnoticed ambiguity *within* Chinchilla's own reported data.
- **(b) Systematic four-family perturbation analysis spanning realistic covariate error sources.** Multiplicative constants, additive constants, systematic bias, and log-normal noise map onto universal sources of covariate error in any scaling law study — not just Chinchilla.
- **(c) Analytical derivations explaining how each perturbation type distorts fitted parameters.** These are mathematical results (Appendix C), not purely empirical observations. They explain *why* the law is robust and *when* it breaks.
- **(d) A proposed diagnostic framework for distinguishing specification artifacts from genuine scaling law failures.** When a practitioner observes that a scaling law's D/N ratio trends with compute, the framework lets them diagnose the cause: a flat vertical shift suggests multiplicative error; a tilt with compute suggests additive error or systematic bias; noisy scatter with a stable median suggests random measurement error.

To our knowledge, no prior scaling law paper has performed systematic covariate sensitivity analysis. We propose this should become standard practice for scaling law papers deriving prescriptions — including work on data repetition (Muennighoff et al.), overtraining (Gadre et al.), inference-optimal scaling (Sardana et al.), and MoE scaling, among others. The perturbation taxonomy and analytical toolkit we provide are not Chinchilla-specific; they apply to any power-law fit.

## Addressing "More In-Depth Theoretical Analysis"

The paper does contain theoretical analysis in Appendix C that we will promote into the main text in the revision. The key results are:

- **Multiplicative error absorption:** We prove that multiplicative specification errors are absorbed into the prefactor via $\hat{A} \approx A \cdot c_m^\alpha$, leaving the scaling exponent — and hence the D/N scaling trend — invariant. This is a mathematical result, not an empirical one.
- **Additive error and variable slope:** Additive errors break the pure power-law form, creating a variable effective slope $N/(N+c_a)$, which the fitter compensates by shifting $\hat{\alpha}$. This analytically explains the direction and monotonicity of the shift. Notably, this is quantitatively consistent with the findings of Pearce & Song and Porian et al. as special cases.
- **Systematic bias exponent rescaling:** We derive $\hat{\alpha} = \alpha / s$ for systematic bias with exponent $s$, and empirically observe $R^2 > 0.999$ agreement between the analytical prediction and the simulation results.

These results provide a unifying framework that reproduces and explains the individual findings of prior work, while extending to perturbation types those works did not consider.

## Data Quality and Diversity (Weakness 2)

The reviewer correctly notes that our Related Work section discusses data quality and diversity, but the analysis focuses on parameter counts. We scope to parameter-count perturbations precisely because they admit clean counterfactual analysis. Parameter counts admit well-defined alternative specifications (e.g., including or excluding embeddings), whereas data quality lacks a canonical metric, making controlled perturbation harder.

That said, extending the sensitivity analysis to data-side perturbations (e.g., data quality filtering thresholds, domain composition) is a natural next step that our perturbation framework could accommodate. We will note this as a concrete future direction in the revised manuscript.

## Key Question 1: Additional Data Points

To our knowledge, no other paper has published per-model pretraining losses at the granularity of Chinchilla's Table A9. Scaling law papers (e.g., LLaMA, DeepSeek) report aggregate trends but not the per-configuration data needed to re-fit the law. We acknowledge this as a limitation: the Chinchilla dataset remains uniquely detailed for this purpose. We agree that validating the framework on additional datasets as they become available is a valuable future direction, and we will note this explicitly.

## Key Question 2: Changing the Functional Form

We deliberately held the functional form fixed to isolate the effect of parameter-counting ambiguity. Changing the functional form simultaneously would confound the analysis — any observed difference could be attributed to either the new form or the new parameter counts, and the two effects could not be disentangled.

Moreover, our diagnostic framework helps distinguish genuine functional-form changes from artifacts of covariate specification. For example, our additive-constant analysis shows that including or excluding embedding parameters creates the *appearance* of a slope change in the D/N ratio, even though the underlying functional form has not changed. Without this kind of controlled analysis, a practitioner who observes such a slope change might incorrectly conclude the functional form needs revision, when the real issue is an accounting choice. This insight — that apparent functional-form changes can be artifacts of specification errors — itself justifies the single-form approach.

## Key Question 3: Theory-Simulation Consistency

Our analytical predictions and simulation results are consistent. Specifically:

- For **multiplicative constants**, we prove $\hat{\alpha} = \alpha$ exactly (Appendix C.2.1), and the simulations confirm invariance of the exponent across the tested range.
- For **systematic bias**, we derive $\hat{\alpha} = \alpha / s$ and empirically observe agreement consistent with the $R^2 > 0.999$ reported above (Section 3.3).
- For **additive constants**, we derive the direction and monotonicity of $\hat{\alpha}$'s shift, and the simulation results match the predicted trend.

Divergence between theory and simulation occurs only at numerical extremes where the fitting procedure encounters numerical issues (e.g., NaN values from overflow). These edge cases lie well outside any realistic perturbation regime and do not affect the paper's conclusions.

## Limitations Section

The reviewer correctly notes that the paper lacks a dedicated limitations section. We will add one in the revision, covering:

1. Our analysis is conditioned on the Chinchilla dataset and fitting methodology — we do not claim extrapolation to modern overtrained regimes.
2. We analyze only parameter-count perturbations, not data-side or optimizer-side sources of error.
3. Our perturbation analysis uses synthetically perturbed covariates, not ground-truth alternative measurements from independent training runs.
4. The Chinchilla dataset is the only publicly available dataset at sufficient granularity for this analysis, limiting our ability to cross-validate on independent data.

## A Broader Perspective

Our results suggest — at least within the Chinchilla regime — that power-law scaling captures regularities robust to moderate covariate misspecification. If a foundational result's covariates can be this wrong for this long without affecting conclusions, it tells us something about what scaling laws actually measure: they appear to be more sensitive to the relative ordering and dynamic range of model sizes than to exact parameter counts. This insight, together with the reusable perturbation framework, is applicable beyond this single robustness check.

## Summary of Planned Revisions

- Promote the key analytical results from Appendix C into the main text to make the theoretical contributions visible alongside the empirical analysis.
- Add a dedicated Limitations section addressing the scope and boundary conditions of our claims.
- Add a "Diagnostic Framework" paragraph to the Discussion, synthesizing the perturbation results into actionable guidance for practitioners.
- Note data-side perturbations and validation on additional datasets as concrete future directions.
- Frame the perturbation methodology as a reusable tool for the scaling law community, not only an audit of Chinchilla.
- Rewrite the introduction to clearly distinguish our contribution from prior concerns and accurately convey the narrative arc.
