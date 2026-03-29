# Rebuttal to Reviewer rRfE

We thank Reviewer rRfE for their thoughtful and encouraging evaluation. As the reviewer notes, surfacing ambiguities in underlying scaling laws is critically important, and we are grateful that the reviewer recognizes the value of this work despite its narrow scope. We address each of the reviewer's questions and suggestions below.

## Abstract and Scope Alignment

The reviewer correctly identifies a tension between the abstract's claim of "renewed confidence in Chinchilla as a durable guide for scaling modern models" and the scoping paragraph that conditions our claims on the Chinchilla dataset and fitting methods. We agree this is a credibility gap, and we will resolve it in the revision by tempering the abstract to match the stated scope.

To clarify our intended narrative: the prior concerns about Chinchilla (wide confidence intervals, cross-approach inconsistencies) motivated us to scrutinize the original data closely. In doing so, we discovered a new, distinct problem -- every single parameter count in Table A9 is ambiguous, with discrepancies as high as 15.2%. Our contribution is to characterize this ambiguity, show it does not undermine the core results, explain why analytically, and generalize the analysis into a reusable sensitivity framework. The revised abstract will accurately reflect this arc rather than overclaiming about modern regimes.

## Code Validation

The reviewer raises an important point about our reliance on the Epoch AI implementation. We have taken concrete steps to validate it:

1. **Reproduction of published results.** We verified correctness by reproducing Epoch AI's published fit parameters using their code as the starting point. Our adapted implementation recovers the same scaling law coefficients.
2. **Independent analytical cross-check.** Our analytical predictions (Appendix C) independently confirm the empirical trends observed in the bootstrap fitting pipeline. For example, we prove that multiplicative specification errors leave the scaling exponent invariant ($\hat{\alpha} = \alpha$), and this is borne out exactly in the simulations. For systematic bias, we derive $\hat{\alpha} = \alpha / s$ and observe $R^2 > 0.999$ empirically. These analytical results serve as an implementation-independent cross-check: if the code had a bug, the analytical and empirical results would diverge.
3. **Consistency across formulas.** All three parameter-counting formulas yield qualitatively consistent scaling law fits, which would be unlikely if the implementation were producing spurious results.

We will note our reliance on Epoch AI's open-source implementation as a limitation and describe the validation steps taken in the revision.

## Limitations Section

We agree that a dedicated Limitations section would strengthen the paper, and we commit to adding one. It will cover:

1. **Conditioned on Chinchilla.** Our empirical results are conditioned on the Chinchilla dataset and Hoffmann et al.'s fitting methodology. We do not claim extrapolation to modern overtrained regimes, different architectures, or different optimizers.
2. **Parameter-count perturbations only.** We scope our sensitivity analysis to parameter-count specification errors because they admit clean counterfactual analysis: the ground-truth architecture is known, so we can construct principled perturbation families. Data-quality perturbations, by contrast, lack canonical ground-truth metrics, making it difficult to define a controlled perturbation. This is a deliberate design choice that strengthens internal validity at the cost of breadth.
3. **Synthetic perturbations, not ground-truth alternatives.** Our four perturbation families are synthetically constructed. While they are motivated by realistic error sources (embedding inclusion, size-dependent miscounting, measurement noise), they are not drawn from actual alternative measurements. The three parameter-counting formulas in Section 2 provide real alternative measurements, but the broader perturbation analysis in Section 3 is synthetic by design.
4. **Code dependency.** Our fitting pipeline builds on Epoch AI's open-source Chinchilla replication code. We will note this dependency explicitly and describe the validation steps taken (reproduction of published coefficients, independent analytical cross-checks, consistency across formulas).

Framing some of these as deliberate scope decisions -- rather than oversights -- will help future readers understand our methodological rationale.

## Differentiation from Porian et al. and Pearce & Song

We appreciate the reviewer's recognition that the paper "makes a clear case for its originality." To further clarify the complementary relationship:

Porian et al. and Pearce & Song each identified specific accounting decisions (head parameters, embedding inclusion, optimizer settings) that explain discrepancies *between* the Chinchilla and Kaplan scaling laws. Our work addresses a different question: we discover a previously unnoticed ambiguity *within* Chinchilla's own reported data, build a systematic sensitivity analysis spanning four perturbation families, and derive analytical explanations that yield a general diagnostic framework.

These contributions are quantitatively consistent. Our additive-constant analysis (Sec. 3.2) predicts that including or excluding embedding parameters shifts $\hat{\alpha}$ in precisely the direction and magnitude that Pearce & Song reported ($\Delta\hat{\alpha} \approx 0.231$), and is similarly consistent with Porian et al.'s reported shifts. In this sense, our framework provides a unifying analytical explanation for what those papers observed empirically.

## Why Were the Parameter Counts Ambiguous?

We can conjecture. The most likely explanation is that the Chinchilla paper internally used a non-standard attention parameter formula. Our "best fit" formula uses a factor of 5 (rather than the standard 4) in the attention block calculation, which could reflect the inclusion of output projections or bias terms not explicitly listed in Table A9's architectural hyperparameters. Such mismatches arise naturally in large collaborative projects when different teams use slightly different conventions.

Based on this experience, we propose that future scaling law papers should adopt a reporting checklist:

1. **State the exact parameter-counting formula** used, with all terms enumerated.
2. **Specify whether embedding parameters are included** in the model size $N$.
3. **Specify whether tied weights are counted once or twice.**
4. **Report both the approximate FLOP estimate** ($C \approx 6ND$) **and the exact FLOP count** if available, so readers can cross-check.

## Chinchilla Continues to Inform Practice

The reviewer asks us to highlight the extent to which Chinchilla continues to inform consequential training runs. We strengthen this argument:

- **DeepSeek** (2024, 2025) explicitly references the Chinchilla compute-optimal point as a baseline when discussing their training token budgets, even when choosing to overtrain relative to it.
- **Llama 3** (Grattafiori et al., 2024) trained on significantly more tokens than the Chinchilla-optimal point would suggest, but calibrated this decision explicitly relative to the Chinchilla baseline, using it to quantify the degree of overtraining.
- **LLaMA 1** (Touvron et al., 2023) directly applied Chinchilla's tokens-per-parameter heuristic in choosing its training configuration.

The pattern is clear: even models that deliberately deviate from compute-optimal training calibrate their decisions *relative to the Chinchilla baseline*. The Chinchilla scaling law remains the standard reference point for reasoning about training efficiency, making the integrity of that baseline a matter of ongoing practical consequence.

## Modern Relevance -- The Correct Framing

While we scope our empirical claims to the Chinchilla regime, we want to highlight why the results matter beyond it. The recent overtraining scaling laws (Gadre et al., 2024; Schaeffer et al., 2025) define their results *as deviations from* the Chinchilla compute-optimal point. If that optimal point were wrong due to parameter misspecification, every overtraining result referencing it would inherit the error. Our paper shows the compute-optimal point is robust to realistic specification errors, which means the mathematical foundation that overtraining and inference-efficient scaling papers build on is solid.

More broadly, our diagnostic framework is regime-independent. The analytical relationships between perturbation type and scaling parameter distortion hold for any power-law fit, whether at Chinchilla scale, overtrained scale, or future scales. Specifically, multiplicative errors preserve exponents while additive errors distort them, as detailed in the perturbation analysis below. Practitioners in any regime can apply the same diagnostic logic to their own fits.

## On "Trivial Experimental Conditions"

The reviewer's summary characterizes our perturbation analysis as providing evidence "under trivial experimental conditions." The results reveal non-trivial structure. The four perturbation families were chosen to span the space of realistic covariate specification errors: percentage-based miscounting (multiplicative), block inclusion/exclusion (additive), size-dependent errors (systematic bias), and measurement noise (log-normal).

The analytical results uncover a previously unknown distinction with concrete diagnostic implications:

- **Multiplicative errors** are absorbed into prefactors via $\hat{A} \approx A \cdot c_m^{\alpha}$, leaving the scaling exponent and hence the D/N trend invariant. The D/N ratio shifts vertically but stays flat.
- **Additive errors** break the pure power-law form, creating a variable effective slope $N/(N + c_a)$, which the fitter compensates by shifting $\hat{\alpha}$. The D/N ratio *tilts* with compute.

This distinction -- that multiplicative errors are benign while additive errors distort the exponent -- was not previously known and gives practitioners a direct way to diagnose the source of anomalous scaling behavior. If a practitioner observes their D/N ratio trending with compute, they can distinguish between a fundamental breakdown of the scaling law and a fixable accounting error based on the *shape* of the trend.

## What Research Does This Paper Enable?

The reviewer asks for a more nuanced future work section. We envision two main directions:

**Applying the perturbation framework to other scaling laws.** Every scaling law paper that derives prescriptive guidance -- Muennighoff et al. (data repetition), Gadre et al. (overtraining), Sardana et al. (inference-efficient scaling), MoE scaling laws -- should assess the sensitivity of its prescriptions to covariate specification errors using our perturbation taxonomy. We provide the analytical toolkit and the four perturbation families; applying them to new settings is a natural and important next step.

**Extending to other covariates.** Our analysis perturbs only the parameter-count covariate. Extending the sensitivity analysis to data-side covariates (e.g., token-count measurement, data quality filtering thresholds, domain composition) and to compute-side covariates (e.g., FLOP estimation methods) would complete the picture. The perturbation framework accommodates these extensions directly.

## Covariate Sensitivity Analysis as Standard Practice

To our knowledge, no prior scaling law paper has performed systematic covariate sensitivity analysis. Papers routinely report confidence intervals from bootstrap resampling -- which captures *statistical* uncertainty from finite data -- but none have stress-tested their conclusions against *specification* uncertainty in the covariates themselves. We propose that covariate sensitivity analysis should become standard practice in scaling law research, and we offer our perturbation taxonomy and analytical toolkit as a template for doing so.

## Summary of Planned Revisions

In response to Reviewer rRfE's feedback, we will:

1. **Temper the abstract** to align with the stated scope, removing overclaims about modern regimes.
2. **Add an explicit code validation statement** in Section 2, noting that we reproduced Epoch AI's published fit parameters and that our analytical predictions independently confirm the empirical trends. We will note our reliance on Epoch AI's open-source implementation and describe the validation steps taken.
3. **Add a Limitations section** covering dataset conditioning, parameter-count-only scope, synthetic perturbation design, and code dependency.
4. **Sharpen the differentiation** from Porian et al. and Pearce & Song, emphasizing the complementary nature of the contributions and the quantitative consistency of our additive-constant analysis with their reported shifts.
5. **Add a conjecture** about the source of the parameter-count ambiguity and a reporting checklist for future scaling law papers.
6. **Strengthen the discussion of Chinchilla's ongoing influence** with concrete examples (DeepSeek, Llama 3, LLaMA 1).
7. **Expand the future work section** to articulate the research enabled by this paper and propose covariate sensitivity analysis as standard practice.
8. **Promote key analytical results from Appendix C** into the main text, including the diagnostic framework for distinguishing perturbation types from observed D/N behavior.
