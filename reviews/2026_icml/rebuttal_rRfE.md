# Rebuttal to Reviewer rRfE

We thank the reviewer for recognizing the importance of surfacing ambiguities in foundational scaling laws.

## Abstract and scope

The abstract overclaims relative to our stated scope. We will temper it — replacing "durable guide for scaling modern models" with language conditioned on the Chinchilla dataset and fitting methods. The intended arc: prior concerns motivated scrutiny, we found a *new* problem (all 50 parameter counts ambiguous, up to 15.2%), showed the conclusions survive, and generalized this into a reusable sensitivity framework.

## Code validation

We adapted Epoch AI's open-source implementation and validated it three ways: (1) reproducing their published fit parameters, (2) independent analytical cross-checks (Appendix B predictions match simulations — e.g., $\hat{\alpha} = \alpha$ exactly for multiplicative errors, $\hat{\alpha} = \alpha/s$ with $R^2 > 0.999$ for systematic bias), and (3) qualitative consistency across all three parameter-counting formulas.

## Limitations

We will add a dedicated section covering: conditioning on Chinchilla's dataset and methods (no extrapolation claims); scope limited to synthetically perturbed parameter counts (not data-side or optimizer-side errors), a principled choice since parameter counts admit well-defined alternative specifications while data quality lacks a canonical metric; and reliance on Epoch AI's implementation.

## Differentiation from Porian et al. and Pearce & Song

Those works identified specific accounting decisions explaining discrepancies *between* Chinchilla and Kaplan. We discover ambiguity *within* Chinchilla's own data, build a systematic sensitivity analysis across four perturbation families, and derive analytical explanations. The analyses are quantitatively consistent: our additive-constant analysis predicts the direction and magnitude of Pearce & Song's reported $\hat{\alpha}$ shift.

## Why the counts were ambiguous

We conjecture Chinchilla internally used a non-standard attention formula (factor of 5 rather than 4, possibly reflecting output projections or bias terms not in Table A9). We propose a reporting checklist for future scaling papers: state the exact counting formula, specify embedding inclusion, specify tied-weight handling, report both $C \approx 6ND$ and exact FLOPs.

## Chinchilla still matters

DeepSeek (2024, 2025) and Llama 3 (Grattafiori et al., 2024) both calibrate their training decisions explicitly relative to the Chinchilla compute-optimal point. LLaMA 1 directly applied the tokens-per-parameter heuristic. Even Czech et al. (2026, arXiv:2603.22339), analyzing Approach 2 biases at Llama 3 frontier scale, use Chinchilla as the reference frame. Overtraining and generative scaling laws (Gadre et al., 2024; Schaeffer et al., 2025) define results as deviations from the Chinchilla-optimal point, so the robustness of that baseline is load-bearing. Our diagnostic framework is regime-independent — the analytical relationships hold for any power-law fit.

## On "trivial experimental conditions"

The perturbation families are methodologically straightforward by construction, but the analytical results reveal non-trivial structure: multiplicative errors preserve the exponent exactly while additive errors distort it in a size-dependent manner. This distinction was previously unknown and has diagnostic value.

In the revision, we will also promote Appendix B results into the main text and expand the future work section to propose covariate sensitivity analysis as standard practice for scaling law research.
