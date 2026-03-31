# Rebuttal to Reviewer EBqg

We appreciate the reviewer's engagement and the acknowledgment that the direction is "meaningful and potentially impactful."

## Practitioner adoption does not test what we test

Adoption tells us the law *works in practice*. It does not tell us whether the prescriptions would change if the parameter counts were defined differently. These are orthogonal questions. The core finding of this paper is that the most influential scaling paper in the field got every single one of its 50 model parameter counts wrong — discrepancies up to 15.2% — and nobody noticed for four years, across hundreds of papers that built on these numbers. A 15% systematic error in a key covariate of a power-law fit should, in principle, distort the exponents and shift the prescriptions. It does not. Understanding *why* is the contribution, and no prior work provides a framework for answering: "how wrong can a scaling law's inputs be before its outputs break?"

## The paper is not purely empirical

Appendix B derives exact functional dependencies of fitted scaling parameters on each perturbation type. We acknowledge these were too easy to miss in the appendix and will promote them to the main text. The key results: (1) multiplicative errors are absorbed into prefactors ($\hat{A} \approx A \cdot c_m^{\alpha}$) while the exponent $\hat{\alpha}$ remains invariant — a proof, not an observation; (2) additive errors create a variable effective slope $N/(N+c_a)$ that shifts $\hat{\alpha}$, analytically reproducing the findings of Pearce & Song and Porian et al. as special cases; (3) systematic bias rescales the exponent as $\hat{\alpha} = \alpha/s$ with $R^2 > 0.999$ empirical agreement.

## Practical guidance

We recommend the Standard Formula (Eqn. 1): it is derived from first principles with no free parameters, it is consistent with Kaplan et al.'s $C \approx 6ND$ convention, and it produces the flattest D/N ratio ($-0.572$ per decade vs $-1.049$ and $-1.248$). While all three formulas preserve the qualitative conclusion, the Standard Formula provides the strongest quantitative stability. The ~20:1 tokens-per-parameter ratio is preserved under all three formulas; what changes is how stable it is across compute budgets.

The analytical results also yield a diagnostic for practitioners: if D/N shifts vertically but stays flat, suspect a multiplicative counting error (exponent unaffected, fix the scaling factor); if D/N tilts with compute, suspect an additive error like embedding inclusion ($\hat{\alpha}$ distorted, investigate size-independent parameter blocks); if D/N becomes noisy but the median holds, suspect random measurement error.

## Why not a modified scaling law?

A modified law is not warranted — the law is correct, the parameter counts were ambiguous. Our diagnostic framework reveals when an *apparent* need to modify the law is actually an artifact of additive specification errors — guidance that helps the community avoid premature modifications.

## Broader contribution

We propose covariate sensitivity analysis as standard practice for scaling law research. Papers by Muennighoff et al. (data repetition), Gadre et al. (overtraining), Sardana et al. (inference-efficient scaling), and MoE scaling work all use covariate definitions that could contain analogous ambiguities. Our perturbation taxonomy and analytical toolkit apply directly. In the revision, we will promote the Appendix B results into the main text, add the diagnostic framework and formula recommendation to the Discussion, and strengthen the narrative arc around the core surprise.
