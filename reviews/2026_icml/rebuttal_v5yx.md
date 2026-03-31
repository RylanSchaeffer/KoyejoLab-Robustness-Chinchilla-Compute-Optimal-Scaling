# Rebuttal to Reviewer v5yx

We thank the reviewer for their constructive feedback. Both concerns are well-taken and we commit to fixing them.

## Introductory framing

The introduction conflates our contribution with prior concerns (wide confidence intervals, cross-approach inconsistencies) that we do not actually address. We are not claiming parameter-count ambiguity caused those issues. The intended arc is: prior concerns motivated close scrutiny of the original data; in doing so, we found a *new* problem — all 50 parameter counts in Table A9 are ambiguous, with discrepancies up to 15.2% — and then showed the conclusions survive, explained why analytically, and generalized this into a reusable sensitivity methodology. The revision will make this arc explicit and temper the abstract to match the stated scope.

## Figure 5 and the meaning of "robust"

We agree the sentence "all four sensitivity analyses demonstrate that Chinchilla's key results withstand sizable perturbations" overclaims. We should have been precise about what is preserved and where things break.

For multiplicative perturbations (Fig. 5, Top Left), what is preserved is the *flat trend* of D/N with compute — not the D/N level. At $c_m = 0.1$, $D/N \approx 200$; at $c_m = 10$, $D/N \approx 2$. Our analytical result predicts exactly this: multiplicative errors are absorbed into the prefactor, preserving the exponent and hence the flatness, while rescaling the ratio. Within the realistic range ($c_m \approx 0.85$–$1.15$, corresponding to the ~15% discrepancy), D/N stays close to 20 and the trend stays flat. The other perturbation types behave similarly: at magnitudes consistent with the actual discrepancy, the ratio stays near 20.

The value of Figure 5 is that it maps the *failure boundary* — where each perturbation type transitions from benign to consequential. In the revision, we will replace the overclaiming sentence, add a shaded band to Figure 5 marking the realistic perturbation range, and explicitly discuss the breakdown threshold for each perturbation type.
