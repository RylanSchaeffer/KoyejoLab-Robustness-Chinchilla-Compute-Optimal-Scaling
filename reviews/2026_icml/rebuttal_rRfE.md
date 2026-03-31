# Rebuttal to Reviewer rRfE

We thank Reviewer rRfE for their thoughtful and encouraging evaluation. As the reviewer notes, surfacing ambiguities in underlying scaling laws is critically important, and we are grateful that the reviewer recognizes the value of this work despite its narrow scope. We address each of the reviewer's questions and suggestions below.

## Abstract and Scope Alignment

The reviewer correctly identifies a tension between the abstract's claim of "renewed confidence in Chinchilla as a durable guide for scaling modern models" and the scoping paragraph that conditions our claims on the Chinchilla dataset and fitting methods. We agree this is a credibility gap, and we will resolve it in the revision by tempering the abstract to match the stated scope.

To clarify our intended narrative: the prior concerns about Chinchilla (wide confidence intervals, cross-approach inconsistencies) motivated us to scrutinize the original data closely. In doing so, we discovered a new, distinct problem -- every single parameter count in Table A9 is ambiguous, with discrepancies as high as 15.2%. Our contribution is to characterize this ambiguity, show it does not undermine the core results, explain why analytically, and generalize the analysis into a reusable sensitivity framework. The revised abstract will accurately reflect this arc rather than overclaiming about modern regimes.

## Code Validation

The reviewer raises an important point about our reliance on the Epoch AI implementation. We have taken concrete steps to validate it:

1. **Reproduction of published results.** We verified correctness by reproducing Epoch AI's published fit parameters using their code as the starting point. Our adapted implementation recovers the same scaling law coefficients.
2. **Independent analytical cross-check.** Our analytical predictions (Appendix B) independently confirm the empirical trends observed in the bootstrap fitting pipeline. For example, we prove that multiplicative specification errors leave the scaling exponent invariant ($\hat{\alpha} = \alpha$), and this is borne out exactly in the simulations. For systematic bias, we derive $\hat{\alpha} = \alpha / s$ and observe $R^2 > 0.999$ empirically. These analytical results serve as an implementation-independent cross-check: if the code had a bug, the analytical and empirical results would diverge.
3. **Consistency across formulas.** All three parameter-counting formulas yield qualitatively consistent scaling law fits, which would be unlikely if the implementation were producing spurious results.

We will note our reliance on Epoch AI's open-source implementation as a limitation and describe the validation steps taken in the revision.

## Limitations Section

We agree that a dedicated Limitations section would strengthen the paper, and we commit to adding one. It will cover:

1. **Conditioned on Chinchilla.** Our empirical results are conditioned on the Chinchilla dataset and Hoffmann et al.'s fitting methodology. We do not claim extrapolation to modern overtrained regimes, different architectures, or different optimizers.
2. **Parameter-count perturbations only; synthetic by design.** We scope to parameter-count specification errors because they admit clean counterfactual analysis with well-defined alternative specifications (e.g., embedding inclusion). The four perturbation families are synthetically constructed, motivated by realistic error sources; the three formulas in Section 2 provide real alternative measurements while Section 3's broader analysis is synthetic. Data-quality perturbations lack canonical ground-truth metrics, making controlled perturbation substantially harder — a principled design choice, not an oversight.
3. **Code dependency.** Our fitting pipeline builds on Epoch AI's open-source Chinchilla replication code. We will note this dependency explicitly and describe the validation steps taken.

## Differentiation from Porian et al. and Pearce & Song

We appreciate the reviewer's recognition that the paper "makes a clear case for its originality." To further clarify the complementary relationship:

Porian et al. and Pearce & Song each identified specific accounting decisions (head parameters, embedding inclusion, optimizer settings) that explain discrepancies *between* the Chinchilla and Kaplan scaling laws. Our work addresses a different question: we discover a previously unnoticed ambiguity *within* Chinchilla's own reported data, build a systematic sensitivity analysis spanning four perturbation families, and derive analytical explanations that yield a general diagnostic framework.

These contributions are quantitatively consistent. Our additive-constant analysis (Sec. 3.2) predicts that including or excluding embedding parameters shifts $\hat{\alpha}$ in precisely the direction and magnitude that Pearce & Song reported ($\Delta\hat{\alpha} \approx 0.231$), and is similarly consistent with Porian et al.'s reported shifts. In this sense, our framework provides a unifying analytical explanation for what those papers observed empirically.

## Why Were the Parameter Counts Ambiguous?

We conjecture that the Chinchilla paper internally used a non-standard attention formula — our "best fit" formula uses a factor of 5 rather than 4, possibly reflecting output projections or bias terms not listed in Table A9. Such silent convention mismatches arise naturally in large collaborative projects. We propose that future scaling law papers adopt a reporting checklist: state the exact counting formula, specify embedding inclusion, specify tied-weight handling, and report both $C \approx 6ND$ and exact FLOPs.

## Chinchilla Continues to Inform Practice

The reviewer asks us to highlight the extent to which Chinchilla continues to inform consequential training runs. We strengthen this argument:

- **DeepSeek** (2024, 2025) explicitly references the Chinchilla compute-optimal point as a baseline when discussing their training token budgets, even when choosing to overtrain relative to it.
- **Llama 3** (Grattafiori et al., 2024) trained on significantly more tokens than the Chinchilla-optimal point would suggest, but calibrated this decision explicitly relative to the Chinchilla baseline, using it to quantify the degree of overtraining.
- **LLaMA 1** (Touvron et al., 2023) directly applied Chinchilla's tokens-per-parameter heuristic in choosing its training configuration.

The pattern is clear: even models that deliberately deviate from compute-optimal training calibrate their decisions *relative to the Chinchilla baseline*. Concurrent work further underscores this: Czech et al. (2026, arXiv:2603.22339) scrutinize Chinchilla's Approach 2 (IsoFLOP parabola fits) at Llama 3 frontier scale, finding systematic biases that imply parameter underallocation — another example of the Chinchilla methodology remaining the reference frame for scaling decisions at the largest scales. (Our paper analyzes Approach 3 following Epoch AI, so the two analyses are complementary.)

## On "Trivial Experimental Conditions"

We respectfully clarify that our perturbation analysis, while methodologically straightforward in construction, reveals non-trivial analytical structure: multiplicative errors preserve the scaling exponent exactly while additive errors distort it in a predictable, size-dependent manner. This distinction was previously unknown and has diagnostic implications for practitioners.

## Modern Relevance and Enabled Research

We scope our empirical claims to the Chinchilla regime, but the results matter beyond it: overtraining and generative scaling laws (Gadre et al., 2024; Schaeffer et al., 2025) define their results as deviations from the Chinchilla compute-optimal point, so the robustness of that baseline is load-bearing. Our diagnostic framework is also regime-independent — the analytical relationships between perturbation type and parameter distortion hold for any power-law fit. In the revision, we will expand the future work section to articulate this: every scaling law paper deriving prescriptions should assess covariate sensitivity, and we provide the taxonomy and analytical toolkit to do so.

## Summary of Planned Revisions

In response to Reviewer rRfE's feedback, we will:

1. **Temper the abstract** to align with the stated scope.
2. **Add code validation statement and Limitations section** in the revision.
3. **Sharpen the differentiation** from Porian et al. and Pearce & Song.
4. **Add a reporting checklist** for future scaling law papers and a conjecture about the ambiguity's origin.
5. **Promote key analytical results from Appendix B** into the main text, including the diagnostic framework.
6. **Expand the future work section** to propose covariate sensitivity analysis as standard practice.
