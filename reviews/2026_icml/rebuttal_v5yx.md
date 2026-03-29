# Rebuttal to Reviewer v5yx

We thank Reviewer v5yx for their careful reading and constructive feedback. We address both concerns below. The low Presentation (1) and Soundness (2) sub-scores appear driven by the two issues below; we commit to fixing both.

## Introductory Framing (Weakness 1)

Reviewer v5yx correctly identifies a gap between our introduction's motivation and our analysis. To clarify: we are *not* claiming that the parameter-count ambiguity is the root cause of the wide confidence intervals or the inconsistencies noted by prior work (e.g., Besiroglu et al., Porian et al., Pearce & Song). Those are separate issues that have been partially addressed by those prior works.

We acknowledge the introduction failed to communicate the intended arc clearly. Our argument is:

1. Chinchilla is foundational but has recently been questioned on multiple fronts. We add a **new, previously unnoticed** reason: every single one of the 50 model parameter counts in Table A9 is ambiguous, with discrepancies as high as 15.2%. This went undetected for four years.
2. One might expect this to undermine the results. It doesn't — the scaling law estimates and the ~20:1 heuristic are essentially unchanged, and under one interpretation (the Standard Formula), the heuristic becomes *more* stable.
3. We explain *why* through analytical results (currently in Appendix C): multiplicative specification errors are absorbed into prefactors while leaving the scaling exponent invariant; additive errors and systematic biases *do* alter the exponent, but in precisely characterizable ways.
4. This yields a diagnostic framework and reusable sensitivity analysis methodology for power-law scaling studies.

We will revise the introduction to make this arc explicit, clearly distinguishing our contribution (a new problem + its resolution + a general framework) from the prior concerns cited in the opening paragraph. We will replace "durable guide for scaling modern models" in the abstract with language scoped to the Chinchilla dataset and fitting methods.

## Clarification on "Robustness" and Figure 5 (Key Question)

We agree with the reviewer that our sentence — "all four sensitivity analyses demonstrate that Chinchilla's key results withstand sizable perturbations" — overclaims relative to what the figures show.

As the reviewer observes, the ratio is robust within the realistic error margin (~15.2%). We should have stated this explicitly. The actual parameter-counting discrepancy we uncovered is at most 15.2%, corresponding to a multiplicative constant of approximately 1.15. At this scale:

- **Multiplicative constant** (Fig. 5, Top Left): the D/N ratio shifts by less than a factor of 2 and remains flat with respect to compute — well within the robust regime.
- **Additive constant** (Fig. 5, Top Right): perturbations on the order of embedding parameters (~16M–160M for these models) preserve the qualitative behavior.
- **Systematic bias and log-normal noise** (Fig. 5, Bottom): at perturbation magnitudes consistent with the observed ~15% discrepancy, the ratio remains close to 20.

To give concrete numbers: the multiplicative constant perturbation preserves the D/N ≈ 20 ratio for constants between approximately 0.1 and 10 (i.e., a full order of magnitude in either direction). The additive constant perturbation preserves qualitative behavior for constants up to roughly ±4×10^7, compared to the smallest model's 42×10^6 parameters.

Figure 5's contribution is to **map the failure boundary**: it shows where each perturbation type transitions from benign to consequential, and our analytical results (Appendix C) explain why. This characterization of when and how the heuristic breaks is more informative than a blanket robustness claim.

In the revision, we will:

1. Replace the overclaiming sentence with: "...withstand perturbations beyond the 15.2% discrepancy we identified, with the nature and threshold of breakdown depending on the perturbation type (see Sections 3.1–3.4)."
2. Add a visual annotation to Figure 5 (e.g., a shaded band at the ~15% realistic perturbation range) so that readers can immediately see the realistic regime is comfortably within the robust zone.
3. Explicitly discuss the threshold at which each perturbation type begins to qualitatively change the results.

## Summary of Planned Revisions

- Rewrite the introduction to clearly distinguish our contribution (new problem + resolution + general framework) from the prior concerns cited in the opening.
- Temper the abstract to match the stated scope.
- Revise the overclaiming sentence about robustness and add explicit discussion of failure thresholds.
- Add visual annotations to Figure 5 indicating the realistic perturbation range.
- Summarize key analytical results from Appendix C in the main text.

We believe these revisions directly address the Presentation and Soundness concerns. Regarding Significance: our contribution is not merely confirming robustness — it is explaining *why* the scaling law is robust, *when* it breaks, and providing a reusable sensitivity analysis methodology for power-law scaling studies. Regarding Originality: this is, to our knowledge, the first identification of the internal parameter ambiguity in Chinchilla, and the first systematic covariate sensitivity analysis for any scaling law.
