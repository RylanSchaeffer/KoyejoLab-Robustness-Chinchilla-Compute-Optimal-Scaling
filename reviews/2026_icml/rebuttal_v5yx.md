# Rebuttal to Reviewer v5yx

We thank the reviewer for the constructive feedback. Both concerns are well-taken — the framing mismatch and the Figure 5 overclaim are presentation failures that obscured the paper's actual contributions. We address both below and then clarify what those contributions are.

## Introductory framing

Agreed: the introduction conflates our contribution with prior concerns (wide confidence intervals, cross-approach inconsistencies) that we do not address. We will fix this. The honest arc is: prior concerns motivated close scrutiny; in doing so, we found a *new* problem — the most influential scaling paper in the field got every single one of its 50 model parameter counts wrong, with discrepancies up to 15.2%, and nobody noticed for four years. One would expect this to matter. It does not. The revision will make this arc explicit and temper the abstract to match the stated scope.

## Figure 5

Agreed: "all four sensitivity analyses demonstrate that Chinchilla's key results withstand sizable perturbations" overclaims. For multiplicative perturbations, what is preserved is the *flat trend* of D/N with compute — not the D/N level ($D/N \approx 200$ at $c_m = 0.1$; $D/N \approx 2$ at $c_m = 10$). Within the realistic range ($c_m \approx 0.85$–$1.15$, the ~15% discrepancy), D/N stays close to 20 and the trend stays flat. Figure 5's actual contribution is mapping the *failure boundary* for each perturbation type. We will add a shaded band marking the realistic range and explicitly discuss breakdown thresholds.

## What the presentation obscured

The framing issues above likely obscured the paper's analytical contributions. The paper is not just an empirical check — Appendix B contains analytical derivations that explain *why* the scaling law is robust to certain perturbations and *when* it breaks. Multiplicative errors are provably absorbed into the prefactor, leaving the exponent invariant. Additive errors distort the exponent in a precisely characterizable way that reproduces the findings of Pearce & Song and Porian et al. as special cases. Systematic bias rescales the exponent as $\hat{\alpha} = \alpha/s$ ($R^2 > 0.999$). These are mathematical results, not empirical observations, and they were buried in the appendix — we will promote them to the main text.

These results also yield a diagnostic framework: the *shape* of an anomalous D/N trend reveals whether the cause is a multiplicative counting error (flat shift), an additive specification error like embedding inclusion (tilt with compute), or random noise (scatter without trend). To our knowledge, no prior scaling law paper has performed systematic covariate sensitivity analysis. We believe this — understanding the conditions under which scaling law prescriptions are and are not sensitive to input specification — is a contribution that the current framing fails to convey.
