# Rebuttal to Reviewer CheF

We appreciate the thorough evaluation and substantive questions.

## Clarification: robustness audit, not scaling law revision

The most influential scaling paper in the field got every single one of its 50 model parameter counts wrong -- discrepancies up to 15.2% -- and nobody noticed for four years. We ask: does correcting them change anything? No. We then ask *why not*, and provide analytical explanations. The three parameter-counting formulas are not alternative scaling laws -- they are three equally valid interpretations of the same architectures. The perturbation analysis generalizes beyond any specific formula to characterize how power-law fits respond to covariate misspecification, suggesting power-law scaling captures something more fundamental than the precise variable plugged in for $N$.

The paper's novel contributions are: (a) first identification of the internal parameter-count ambiguity within Chinchilla's own data (prior works addressed discrepancies *between* Chinchilla and Kaplan); (b) systematic four-family perturbation analysis spanning realistic covariate error sources; (c) analytical derivations (Appendix B) explaining *why* the law is robust and *when* it breaks; and (d) a diagnostic framework for distinguishing specification artifacts from genuine scaling law failures. To our knowledge, no prior scaling law paper has performed systematic covariate sensitivity analysis. Czech et al. (2026, arXiv:2603.22339) independently analyze Chinchilla's Approach 2 at Llama 3 frontier scale -- complementary to our Approach 3 analysis.

## Theoretical analysis

Appendix B (to be promoted to the main text) proves: multiplicative errors are absorbed into the prefactor via $\hat{A} \approx A \cdot c_m^\alpha$, leaving the exponent invariant; additive errors create a variable effective slope $N/(N+c_a)$ that shifts $\hat{\alpha}$ (quantitatively consistent with Pearce & Song and Porian et al.); systematic bias rescales the exponent as $\hat{\alpha} = \alpha/s$ with $R^2 > 0.999$.

## Data quality and diversity

We scope to parameter-count perturbations because they admit clean counterfactual analysis -- parameter counts have well-defined alternative specifications, whereas data quality lacks a canonical metric. This is a principled methodological choice. Extending to data-side perturbations is a natural next step we will note in the revision.

## Key questions

**Additional data (Q1):** No other paper publishes per-model losses at the granularity of Table A9. Czech et al. (2026) recently digitized Llama 3 IsoFLOP loss curves, which could serve as an independent test bed in future work.

**Changing functional form (Q2):** We held the form fixed deliberately -- changing it simultaneously would confound the analysis. Our framework actually helps here: the additive-constant analysis shows that including/excluding embeddings creates the *appearance* of a slope change even though the functional form is unchanged, guarding against premature functional-form revisions.

**Theory-simulation consistency (Q3):** The analytical predictions and simulations agree: $\hat{\alpha} = \alpha$ exactly for multiplicative errors, $\hat{\alpha} = \alpha/s$ for systematic bias ($R^2 > 0.999$), correct direction and monotonicity for additive errors. Divergence occurs only at perturbation magnitudes far beyond any realistic scenario.

## Limitations

We will add a dedicated section covering: conditioning on Chinchilla's dataset and fitting methodology; scope limited to parameter-count perturbations; synthetic perturbation design; and reliance on Epoch AI's implementation (validated by reproducing their published coefficients and by independent analytical cross-checks).
