# Rebuttal to Reviewer v5yx

We thank Reviewer v5yx for their careful reading and constructive feedback. We address both concerns below and commit to fixing both in the revision.

## Introductory Framing (Weakness 1)

Reviewer v5yx correctly identifies a gap between our introduction's motivation and our analysis. To clarify: we are *not* claiming that the parameter-count ambiguity is the root cause of the wide confidence intervals or the inconsistencies noted by prior work (e.g., Besiroglu et al., Porian et al., Pearce & Song). Those are separate issues that have been partially addressed by those prior works.

We acknowledge the introduction failed to communicate the intended arc clearly. Our argument is:

1. Chinchilla is foundational but has recently been questioned on multiple fronts. We add a **new, previously unnoticed** reason: every single one of the 50 model parameter counts in Table A9 is ambiguous, with discrepancies as high as 15.2%. This went undetected for four years.
2. One might expect this to undermine the results. It doesn't — the scaling law estimates and the ~20:1 heuristic are essentially unchanged, and under one interpretation (the Standard Formula), the heuristic becomes *more* stable.
3. Analytical results (Appendix B) explain *why*: multiplicative errors preserve the exponent; additive errors and biases distort it in precisely characterizable ways.
4. These results yield a diagnostic framework and reusable sensitivity methodology for any power-law scaling study.

We will revise the introduction to make this arc explicit, clearly distinguishing our contribution (a new problem + its resolution + a general framework) from the prior concerns cited in the opening paragraph. We will replace "durable guide for scaling modern models" in the abstract with language scoped to the Chinchilla dataset and fitting methods.

## Clarification on "Robustness" and Figure 5 (Key Question)

We agree with the reviewer that our sentence — "all four sensitivity analyses demonstrate that Chinchilla's key results withstand sizable perturbations" — overclaims relative to what the figures show.

As the reviewer observes, the ratio is robust within the realistic error margin (~15.2%). We should have stated this explicitly. The actual parameter-counting discrepancy we uncovered is at most 15.2%, corresponding to a multiplicative constant of approximately 1.15. At this scale:

To be precise about what "robust" means for each perturbation type: the **multiplicative constant** (Fig. 5, Top Left) preserves the *flat trend* of D/N with respect to compute for any $c_m$ — but shifts the D/N level itself ($D/N \approx 200$ at $c_m = 0.1$; $D/N \approx 2$ at $c_m = 10$). This is exactly what our analytical result predicts: multiplicative errors are absorbed into the prefactor, preserving the exponent and hence the flatness, while rescaling the ratio. Within the realistic range ($c_m \approx 0.85$–$1.15$, i.e., the ~15% discrepancy), the D/N level stays close to 20 and the trend remains flat. For the **additive constant** (Fig. 5, Top Right), perturbations on the order of embedding parameters (~16M–160M) preserve qualitative behavior. For **systematic bias and log-normal noise** (Fig. 5, Bottom), perturbation magnitudes consistent with the ~15% discrepancy keep the ratio close to 20.

Figure 5's contribution is to **map the failure boundary**: it shows where each perturbation type transitions from benign to consequential, and our analytical results (Appendix B) explain why.

In the revision, we will:

1. Replace the overclaiming sentence with: "...withstand perturbations at and beyond the scale of the 15.2% discrepancy we identified, with the nature and threshold of breakdown depending on the perturbation type (see Sections 3.1–3.4)."
2. Add a visual annotation to Figure 5 (e.g., a shaded band at the ~15% realistic perturbation range) so that readers can immediately see the realistic regime is comfortably within the robust zone.
3. Explicitly discuss the threshold at which each perturbation type begins to qualitatively change the results.

## Summary of Planned Revisions

- Rewrite the introduction to clearly distinguish our contribution (new problem + resolution + general framework) from the prior concerns cited in the opening.
- Temper the abstract to match the stated scope.
- Revise the overclaiming sentence about robustness and add explicit discussion of failure thresholds.
- Add visual annotations to Figure 5 indicating the realistic perturbation range.
- Summarize key analytical results from Appendix B in the main text.

We believe these revisions directly address the reviewer's concerns.
