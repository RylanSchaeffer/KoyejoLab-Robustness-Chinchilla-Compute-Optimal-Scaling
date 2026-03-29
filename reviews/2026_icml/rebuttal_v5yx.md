# Rebuttal to Reviewer v5yx

We thank Reviewer v5yx for their careful reading and constructive feedback. We are glad the reviewer finds the parameter-count ambiguity "a valuable observation" and the empirical approach "well-structured." We address both concerns below.

## Introductory Framing (Weakness 1)

Reviewer v5yx correctly identifies a gap between our introduction's motivation and our analysis. To clarify: we are *not* claiming that the parameter-count ambiguity is the root cause of the wide confidence intervals or the inconsistencies noted by prior work (e.g., Besiroglu et al., Porian et al., Pearce & Song). Those are separate issues that have been partially addressed by those prior works.

Our intended narrative arc is different, and we acknowledge we failed to communicate it clearly:

1. Chinchilla is foundational — the field's standard reference for compute-optimal scaling.
2. It has recently been questioned on multiple fronts (wide CIs, internal inconsistencies between its three approaches, discrepancies with Kaplan et al.).
3. We add a **new, previously unnoticed** reason to question it: *every single one* of the 50 model parameter counts in Chinchilla's Table A9 — the very covariates on which the scaling law fits rest — is ambiguous. Three plausible interpretations of these counts exist, with definitional discrepancies as high as 15.2%. This went undetected for four years.
4. The natural expectation is that this should undermine the results. It doesn't. The scaling law estimates and the ~20:1 tokens-per-parameter heuristic are essentially unchanged — and under one interpretation, the heuristic becomes *more* stable.
5. We explain *why* through analytical results (currently in Appendix C): multiplicative specification errors are absorbed into prefactors while leaving the scaling exponent invariant; additive errors and systematic biases *do* alter the exponent, but in precisely characterizable ways.
6. This yields a general diagnostic framework and a reusable sensitivity analysis methodology applicable to any power-law scaling study.

We will revise the introduction to make this arc explicit, clearly distinguishing our contribution (a new problem + its resolution + a general framework) from the prior concerns cited in the opening paragraph. We will also temper the abstract to align with our stated scope — replacing "durable guide for scaling modern models" with language that accurately reflects our claims, which are conditional on the Chinchilla dataset and fitting methods.

## Clarification on "Robustness" and Figure 5 (Key Question)

Reviewer v5yx is correct that under extreme perturbations, the curves in Figure 5 diverge from the D/N = 20 baseline. We do not claim the scaling law is universally indestructible, and we appreciate the reviewer pointing out that our sentence — "all four sensitivity analyses demonstrate that Chinchilla's key results withstand sizable perturbations" — overclaims relative to what the figures show.

As the reviewer themselves observes, "the ratio is robust within the realistic error margin (e.g., the ~15.2% discrepancy found in Section 2)." That is exactly right, and we should have stated this explicitly. The actual parameter-counting discrepancy we uncovered is at most 15.2%, corresponding to a multiplicative constant of approximately 1.15. At this scale:

- **Multiplicative constant** (Fig. 5, Top Left): the D/N ratio shifts by less than a factor of 2 and remains flat with respect to compute — well within the robust regime.
- **Additive constant** (Fig. 5, Top Right): perturbations on the order of embedding parameters (~16M–160M for these models) preserve the qualitative behavior.
- **Systematic bias and log-normal noise** (Fig. 5, Bottom): at perturbation magnitudes consistent with the observed ~15% discrepancy, the ratio remains close to 20.

The contribution of Figure 5 is not to claim robustness everywhere — it is to **map the failure boundary**. The figure shows precisely where each perturbation type transitions from benign to consequential, and our analytical results (Appendix C) explain *why* each transition occurs. We believe this characterization of when and how the heuristic breaks is more informative than a blanket robustness claim.

In the revision, we will:

1. Replace the overclaiming sentence with: "...withstand perturbations well beyond the 15.2% discrepancy we identified, with the nature and threshold of breakdown depending on the perturbation type (see Sections 3.1–3.4)."
2. Add a visual annotation to Figure 5 (e.g., a shaded band at the ~15% realistic perturbation range) so that readers can immediately see the realistic regime is comfortably within the robust zone.
3. Explicitly discuss the threshold at which each perturbation type begins to qualitatively change the results.

## Summary of Planned Revisions

- Rewrite the introduction to clearly distinguish our contribution (new problem + resolution + general framework) from the prior concerns cited in the opening.
- Temper the abstract to match the stated scope.
- Revise the overclaiming sentence about robustness and add explicit discussion of failure thresholds.
- Add visual annotations to Figure 5 indicating the realistic perturbation range.
- Promote the key analytical results from Appendix C into the main text to make the mechanistic explanations more visible.
