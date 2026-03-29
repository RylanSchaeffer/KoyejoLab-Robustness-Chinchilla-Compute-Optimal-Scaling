# REBUTTAL_STRATEGY.md

Current scores: **3 (EBqg), 3 (CheF), 4 (rRfE), 3 (v5yx)** — Average 3.25, one weak accept, three weak rejects.

Target: Move at least two weak-rejects to accept (5+), hold rRfE at accept or above, and raise the overall impression to oral tier.

---

## Synthesizing Reviewer Objections

### Reviewer-by-reviewer summary (preserving each reviewer's distinct position)

**Reviewer EBqg — Weak Reject (3), Confidence 2 (low)**

EBqg's core objection is significance: the paper "primarily justifies the robustness of the Chinchilla scaling law, which has already been empirically validated many times in practice (e.g., it was adopted in training earlier series of LLaMA models)." They want either a new/modified scaling law or new practical guidance (e.g., a recommendation on the 20-tokens-per-parameter rule). They also ask which parameter-counting formula should be preferred — the paper presents three and effectively says "doesn't matter," which the reviewer finds unsatisfying. In their limitations section, they call the paper "purely an empirical study" that "does not provide practical guidance for LLM development" — notable because the paper *does* have theoretical analysis in the appendix, suggesting EBqg either missed it or found it insufficient.

On the positive side, EBqg acknowledges the paper is "generally well written and well motivated," calls investigating scaling law robustness "meaningful and potentially impactful," and finds the parameter-counting observation "interesting." Their confidence is 2 (low), meaning they admit they may not have understood the central parts — this makes them more persuadable.

Sub-scores: Soundness 3, Presentation 3, Significance 2, Originality 3.

**Reviewer CheF — Weak Reject (3), Confidence 3**

CheF's framing is that the "main contribution of this paper can be seen as a new fit to the data in Hoffmann et al. (2022), revising the original law's function and performing robustness tests" — characterizing it as technically sound but narrow in scope. They raise two specific concerns beyond significance: (1) "the authors have not considered some practical issues, such as data quality and diversity mentioned in their related work" — pointing out that our own Related Work section discusses data quality/diversity as relevant to scaling, but our analysis ignores these factors; and (2) "a more in-depth theoretical analysis could be conducted" given the data size and equation form.

Their key questions ask about additional data from subsequent papers, whether we considered changing the functional form of the scaling law, and whether there are inconsistencies between the theoretical analysis and simulation results. Their limitations note says the paper lacks a dedicated limitations section.

On the positive side, CheF acknowledges the paper provides "a detailed discussion of the Chinchilla scaling laws" and that "the revised formula performs better and may be more generalizable," and that the perturbation experiments "verify the stability" of the scaling laws. Note: CheF's summary characterizes the paper as trying to "adjust the Chinchilla scaling laws," which is a slight misread — we analyze robustness, not adjust the laws.

CheF does *not* mention Porian et al. or Pearce & Song. Their novelty concern is general, not about overlap with specific prior work.

Sub-scores: Soundness 3, Presentation 3, Significance 2, Originality 2 (tied with v5yx for the lowest originality score across all reviewers).

**Reviewer rRfE — Weak Accept (4), Confidence 2 (low)**

rRfE is the most sympathetic reviewer and the only one who gave an accept. Crucially, they explicitly state: "While the scope of this work is narrow, surfacing ambiguities in underlying scaling laws is so important." They also say the paper "makes a clear case for its originality." These are strong positive anchors.

However, rRfE has several specific asks: (1) They want to know if the authors "taken concrete steps to validate this implementation" (the Epoch AI code). (2) They want "a clearer limitations section" — or alternatively, "the authors' commentary on why their work does not suffer from sufficient limitations" (giving us two options). (3) They want the authors to "highlighting the extent to which Chinchilla continues to inform consequential training runs" — this is a distinct ask from "modern relevance"; they want evidence that practitioners *still use* Chinchilla-style scaling, which is a citation/practice argument. (4) They ask the authors to "clarify how their work complements" Porian et al. (2024) and Pearce & Song (2024) — but this is a request for *clarification*, not a challenge to originality.

rRfE also notes a tension between the abstract ("durable guide for scaling modern models") and the scope paragraph ("conditional on the Chinchilla dataset and fitting methods"), and asks whether we can conjecture why the parameter counts were ambiguous in the first place and how future papers can avoid this.

Their summary describes the results as providing evidence "under trivial experimental conditions" — a somewhat dismissive characterization of the perturbation analysis that suggests they view it as straightforward.

Sub-scores: Soundness 3, Presentation 3, Significance 2, Originality 3.

**Reviewer v5yx — Weak Reject (3), Confidence 3**

v5yx's position is distinct from EBqg and CheF. They do *not* primarily question the paper's significance. In their strengths, they explicitly call the ambiguity finding "a valuable observation" and the empirical approach "well-structured." Their complaint is focused and specific: **the introductory framing over-promises**.

Their sole weakness is: the introduction cites "recent concerns regarding wide confidence intervals and inconsistent results" and promises "renewed confidence," but "the paper does not adequately bridge the gap between these specific concerns and the proposed analysis." They cannot tell if we are claiming the parameter ambiguity caused those prior concerns or if we are addressing something orthogonal. "If the parameter ambiguity is unrelated to those specific concerns, the introductory framing might be over-promising the scope of the resolution this paper provides."

Their key question is about Figure 5: "a direct visual inspection of Figure 5 seems to contradict this statement" that results "withstand sizable perturbations." They want explicit criteria for "withstand" and "sizable," and want to know "the threshold at which the heuristic fails." Importantly, v5yx themselves note that "the ratio is robust within the realistic error margin (e.g., the ~15.2% discrepancy found in Section 2)" — they already see the answer, they just want the paper to say it clearly.

**v5yx gave Presentation: 1 (poor)** — the single worst sub-score in the entire review set. This is a red flag that the reviewer found the paper actively confusing or misleading to read, almost certainly driven by the framing mismatch. They also gave **Soundness: 2 (fair)**, likely driven by the Figure 5 / overclaimed robustness issue. Because v5yx has a single focused complaint (framing) and already acknowledges the core finding as "valuable," fixing the framing and the overclaiming could swing this reviewer substantially.

Sub-scores: Soundness 2, Presentation 1, Significance 2, Originality 2.

---

### Thematic concerns across reviewers

With the per-reviewer positions established, the cross-cutting themes are:

#### 1. Incremental significance / "why does this paper exist?" (EBqg, CheF)

Two reviewers — EBqg and CheF — genuinely question whether the contribution warrants publication. EBqg sees it as confirming what practitioners already know; CheF sees it as a narrow refit plus robustness checks. Both want either new guidance, new laws, or deeper theory.

Note: v5yx is *not* in this camp. Their concern is framing, not significance. rRfE explicitly endorses the importance of the ambiguity finding.

**Diagnosis:** The manuscript frames the contribution as *validation* ("Chinchilla is robust"). It should frame as *mechanistic understanding* (why robust, when it breaks, how to tell the difference) and *methodological contribution* (a reusable sensitivity analysis framework for any scaling law). The paper also undersells its most surprising finding: all 50 parameter counts were wrong for four years, yet the conclusions survive. The reader needs to feel the vertigo before receiving the punchline. EBqg's "purely empirical" characterization also suggests the theoretical analysis in the appendix was missed — it needs to be promoted into the main text.

#### 2. Introductory framing over-promises and misleads (v5yx; related concern from rRfE)

v5yx's primary complaint, and likely the driver of their Presentation: 1 score. The introduction invokes "wide confidence intervals and inconsistent results" but the analysis addresses a distinct problem (parameter ambiguity). We are, in fact, addressing something different under the same rhetorical umbrella, but the paper doesn't make this clear.

rRfE raises a related tension: the abstract claims guidance for "modern models" but the scope section says "conditional on Chinchilla dataset."

**Diagnosis:** The introduction needs to be honest: prior concerns motivated close scrutiny; in doing so we discovered a *new* problem (internal parameter ambiguity) that is distinct from but related to those concerns. The rhetorical arc should be: "Chinchilla has been questioned for several reasons. We add another reason it *should* have been questioned — the covariates themselves were wrong — and then show that even this doesn't break it." The abstract scope needs to match the stated scope.

#### 3. Figure 5 contradicts "robust" claims (v5yx)

v5yx directly challenges the main conclusion: Figure 5's curves "fan out drastically" under perturbation, yet the text says results "withstand sizable perturbations." They want the threshold at which the heuristic fails. Notably, v5yx themselves observe that robustness holds "within the realistic error margin (e.g., the ~15.2% discrepancy)" — they already see the answer but want it stated explicitly. This likely contributed to their Soundness: 2 score.

**Diagnosis:** The sentence "all four sensitivity analyses demonstrate that Chinchilla's key results withstand sizable perturbations" is an overclaim that the paper's own figures contradict. The rebuttal should acknowledge extreme perturbations break the ratio, quantify the realistic regime (~15% is well within the robust zone), and reframe Figure 5's contribution as *mapping the failure boundary* rather than claiming invincibility. Adding a visual annotation (e.g., shaded band at 15%) to the figure would directly address this.

#### 4. Missing practical guidance and limitations (EBqg, CheF, rRfE)

Specific asks, enumerated:

- **EBqg:** Which formula to prefer? Does this suggest a new/modified scaling law? (EBqg calls the paper "purely an empirical study" — missing the appendix theory.)
- **CheF:** Additional data from subsequent papers? Changing the functional form? Data quality/diversity considerations (specifically flagged as mentioned in our Related Work but not addressed in the analysis)?
- **rRfE:** Limitations section (or argument for why none is needed). Code validation of the Epoch AI implementation. Evidence that Chinchilla *continues to inform* consequential training runs (a citation/practice question, distinct from theoretical extrapolation to modern regimes).

**Diagnosis:** Recommend the Standard Formula (first-principles, flattest D/N slope, consistent with Kaplan). Add a Limitations paragraph. State code validation explicitly (we reproduce Epoch AI's numbers; our analytical predictions independently confirm empirical trends). Acknowledge data boundary (Chinchilla is the only public dataset with per-model losses at this granularity). For rRfE's "continues to inform" question, cite concrete examples of Chinchilla-informed training decisions (LLaMA, etc.).

#### 5. Differentiation from Porian et al. and Pearce & Song (rRfE)

rRfE — and *only* rRfE — explicitly asks how this work "complements" Porian et al. (2024) and Pearce & Song (2024). This is a request for clarification, not a challenge: rRfE already says the paper "makes a clear case for its originality." CheF has a general novelty concern ("can be seen as a new fit") but does not mention these specific papers.

**Diagnosis:** The differentiation is: Porian and Pearce each identified *specific* accounting decisions explaining discrepancies *between* Chinchilla and Kaplan. We discover an *internal* ambiguity within Chinchilla's own Table A9, build a *systematic* sensitivity analysis across four perturbation families, and derive analytical explanations yielding a general diagnostic framework. Our additive-constant analysis subsumes and explains both Porian's and Pearce's findings as special cases (manuscript Sec. 3.2).

---

## Prioritized Rebuttal Plan

### Strategic overview: which reviewers are movable?

- **v5yx is the most movable.** They have a single focused complaint (framing mismatch) and already call the ambiguity finding "a valuable observation" and the approach "well-structured." Their Presentation: 1 score is almost certainly driven by the framing issue; their Soundness: 2 is likely driven by the Figure 5 overclaim. Fix those two things and this reviewer could swing from 3 to 5.
- **EBqg is next most movable.** Low confidence (2) means they admit uncertainty about their own assessment. They acknowledge the paper is well-written and the direction is meaningful. Their core objection is "where's the new guidance?" — which we can directly answer (formula recommendation, diagnostic framework). Their "purely empirical" characterization suggests they missed the appendix theory — correctable.
- **CheF is the hardest to move.** Higher confidence (3), tied-lowest originality score, multiple concerns (novelty, data quality, deeper theory). But they respond to theoretical depth — promoting the appendix material helps.
- **rRfE needs to be held/strengthened.** Already a weak accept, sympathetic to the core contribution. Concrete responses to their specific asks (code validation, limitations, Porian/Pearce differentiation, Chinchilla's ongoing influence) should solidify or raise their score.

---

### Priority 1: Fix the introductory framing and claims calibration (targets v5yx directly)

This is the single highest-leverage fix. v5yx is the most movable reviewer with a single focused complaint, and their Presentation: 1 — the worst sub-score in the entire review set — is almost certainly driven by this issue. rRfE raises a related abstract/scope tension.

v5yx's sole weakness: the introduction cites "wide confidence intervals and inconsistent results" but the analysis addresses parameter ambiguity, a different problem. We are, in fact, addressing something different under the same rhetorical umbrella, but the paper doesn't say this. The reader feels bait-and-switched.

**What to do in the rebuttal text:**
- "Reviewer v5yx correctly identifies that our analysis addresses parameter counting ambiguity, not the wide confidence intervals or approach-3 inconsistencies noted by prior work. We will revise the introduction to be explicit: the prior concerns motivated us to scrutinize Chinchilla closely, and in doing so we discovered a *new* problem — internal parameter ambiguity — that is distinct from but related to those prior issues. (For instance, Pearce & Song showed that embedding inclusion shifts $\hat{\alpha}$ by 0.231, which is quantitatively consistent with our additive-constant analysis.) Our contribution is to resolve this new problem and build a general sensitivity framework on top of it."
- Temper the abstract: replace "renewed confidence in Chinchilla as a durable guide for scaling modern models" with language that matches the stated scope.

**What to do in the revised manuscript:**
- Rewrite the second paragraph of the introduction to create the honest arc: prior concerns exist → they motivated close scrutiny → we found a *new* issue (parameter ambiguity) → we show it doesn't matter → we explain why → we generalize.
- Align abstract scope with the Scope paragraph.

**Addresses:** v5yx (sole weakness, likely driver of Presentation: 1), rRfE (abstract/scope credibility gap).

### Priority 2: Address Figure 5 overclaim (targets v5yx's Soundness: 2)

v5yx's other concern, and likely the driver of their Soundness: 2 score. Notably, v5yx themselves observe that robustness holds "within the realistic error margin (e.g., the ~15.2% discrepancy found in Section 2)" — they already see the answer but want the paper to say it explicitly rather than making the blanket claim that results "withstand sizable perturbations."

**What to do in the rebuttal text:**
- Acknowledge: "Reviewer v5yx is correct that under extreme perturbations the curves in Figure 5 diverge from D/N = 20. We do not claim the law is universally indestructible."
- Quantify: "The actual parameter counting discrepancy we uncovered is at most 15.2%, corresponding to a multiplicative constant of ~1.15. At this scale, Figure 5 (Top Left) shows the D/N ratio shifts by less than a factor of 2 and remains flat — well within the robust regime."
- Reframe: "The value of Figure 5 is not to claim robustness everywhere, but to *map the failure boundary*. The figure shows precisely where each perturbation type transitions from benign to consequential. Knowing that boundary is the contribution — not claiming it doesn't exist."

**What to do in the revised manuscript:**
- Revise "all four sensitivity analyses demonstrate that Chinchilla's key results withstand sizable perturbations" to something precise: "...withstand perturbations well beyond the 15.2% discrepancy we identified, with the nature and threshold of breakdown depending on the perturbation type (see Sections 3.1–3.4)."
- Add a shaded vertical band or annotation in Figure 5's caption/plots indicating the realistic perturbation range (~15%), so the reader can immediately see that the realistic regime is comfortably within the robust zone.

**Addresses:** v5yx (key question, likely driver of Soundness: 2).

### Priority 3: Control the narrative — make the reader feel the surprise before the punchline (targets EBqg, CheF)

The paper's most surprising finding is undersold. Every single parameter count in the most influential scaling paper was wrong — definitional discrepancies as high as 15.2% — and nobody noticed for years. If a reader sees "15.2% discrepancy" and thinks "that's small, who cares," then the robustness result feels obvious and the paper reads as confirmatory. But if the reader first internalizes "wait, *all 50 parameter counts* in the foundational scaling paper were ambiguous, and the field built on this for four years without noticing" — then the fact that the conclusions survive anyway becomes genuinely surprising.

This reframing matters most for EBqg and CheF, who see the paper as confirming what practitioners already believe.

**What to do in the rebuttal text:**
- Open with: "We want to highlight what we believe is the core surprise of this paper, which we may have undersold: every single model parameter count in Chinchilla's Table A9 — the dataset on which the field's compute-optimal scaling prescriptions rest — was ambiguous, with discrepancies reaching 15.2%. This went unnoticed for four years. The natural expectation is that this should matter. It doesn't. Understanding *why* it doesn't is the intellectual contribution of this paper."

**What to do in the revised manuscript:**
- Restructure the introduction's narrative arc: (1) Chinchilla is foundational, (2) it has been questioned, (3) we add a *new* reason to question it — the covariates themselves were wrong, (4) yet the conclusions are unassailable, (5) we explain why, and (6) this has implications for how the field should think about scaling law sensitivity in general.
- Linger on the finding before resolving it. Currently the paper moves from "parameters were ambiguous" to "but results are robust" in a few sentences. Give the reader time to sit with the problem.

**Addresses:** EBqg (significance — reframes from "confirmatory" to "surprising"), CheF (significance — same reframing).

### Priority 4: Promote the analytical results from appendix to main text (targets EBqg, CheF)

The analytical results currently buried in Appendix C are the intellectual core of the paper:
- Multiplicative errors are absorbed into prefactors via $\hat{A} \approx A c_m^\alpha$, leaving the exponent and hence the D/N scaling trend invariant. This is a *mathematical proof*, not an empirical observation.
- Additive errors break the pure power-law form, creating a variable effective slope $N/(N+c_a)$, which the fitter compensates by shifting $\hat{\alpha}$. This *exactly explains* the Pearce & Song and Porian et al. results quantitatively.
- Systematic bias rescales the exponent as $\hat{\alpha} = \alpha/s$, with $R^2 > 0.999$ empirical fit.

These results give practitioners a diagnostic tool: if you observe a D/N ratio trending with compute, you can distinguish between a fundamental breakdown and a fixable accounting error based on the *shape* of the trend.

This directly addresses EBqg's characterization of the paper as "purely an empirical study" — either they missed the appendix theory or found it insufficient because it wasn't foregrounded. It also addresses CheF's ask for "more in-depth theoretical analysis."

**What to do in the rebuttal text:**
- "We note that the paper does contain theoretical analysis (Appendix C) deriving the exact functional dependence of each fit parameter on each perturbation type — these are not purely empirical observations. We will promote the key analytical results into the main text in the revision."
- "When a practitioner observes that a scaling law's D/N ratio trends with compute, our framework lets them diagnose the cause. If the ratio shifts vertically but stays flat: suspect a multiplicative error. If the ratio tilts (slopes with compute): suspect an additive error or systematic bias. This is directly actionable guidance."

**What to do in the revised manuscript:**
- Promote the key analytical results from the appendix into the main text (at least the key equations and the diagnostic interpretation).
- Add a paragraph in the Discussion titled "A Diagnostic Framework" that synthesizes the perturbation results into a decision tree:
  - D/N shifts up/down but stays flat → multiplicative error → fix by correcting the scaling factor
  - D/N trends upward with C → positive additive error (e.g., embeddings included) → fix by excluding the additive component
  - D/N trends downward with C → negative additive error or systematic bias with $s > 1$ → investigate size-dependent counting
  - D/N becomes noisy but median holds → random measurement error → increase bootstrap samples

**Addresses:** EBqg (corrects "purely empirical" misperception; provides practical guidance), CheF ("more in-depth theoretical analysis"; practical significance).

### Priority 5: Elevate the meta-scientific insight and the methodological contribution (targets EBqg, CheF, rRfE)

The deeper finding is not just "Chinchilla is robust" — it's "what does it mean that a power law doesn't care about 15% errors in its input variable?" This borders on saying that scaling laws capture a coarser-grained regularity than we thought. The specific variable we plug in for N matters less than the field assumes. That's a statement about the *nature* of scaling laws, not just about Chinchilla.

Simultaneously, the perturbation framework is a general-purpose tool that any scaling law paper should adopt. The four perturbation types map onto universal sources of covariate error:
- Multiplicative constants → percentage-based miscounting (any architecture)
- Additive constants → inclusion/exclusion of parameter blocks (embeddings, heads, adapters)
- Systematic bias → errors that scale with model size
- Log-normal noise → measurement uncertainty, lottery-ticket-style variability

No prior scaling law paper has performed this kind of sensitivity analysis. We are not just auditing Chinchilla — we are establishing that covariate sensitivity analysis should be standard practice for scaling law research, and providing the template.

**What to do in the rebuttal text:**
- "Our paper makes a meta-scientific observation: if a foundational result's covariates can be this wrong without affecting conclusions, it tells us something deep about what scaling laws actually measure. The law is fitting something more fundamental than the precise variable we plug in for N. We believe this insight — that power-law scaling captures a coarser regularity than the specific covariate definition — is the most important takeaway."
- "Furthermore, our perturbation framework is not Chinchilla-specific. We propose that sensitivity analysis of covariate specification should become standard practice in scaling law research. Our four perturbation types (multiplicative, additive, systematic bias, noise) map onto universal error sources, and the analytical toolkit we provide (Appendix C) applies to any power-law fit. No prior scaling law paper has performed this kind of analysis."

**What to do in the revised manuscript:**
- Add a paragraph to the Discussion making the meta-scientific point explicit: scaling laws are more robust to covariate misspecification than the field assumes, which suggests they capture coarser regularities.
- Reframe the Future Directions paragraph from "we could do this for other scaling laws" (a to-do item) to "the field should do this for all scaling laws" (a methodological recommendation). Every paper that derives prescriptive guidance from a power-law fit should stress-test its conclusions against specification errors in the covariates, using the perturbation taxonomy we provide.

**Addresses:** EBqg ("what additional contribution does this paper provide"), CheF (significance, novelty), rRfE (modern relevance — the framework is regime-independent even though the empirical claims are scoped to Chinchilla).

### Priority 6: State a clear recommendation for parameter counting (targets EBqg)

EBqg asks directly: "which parameter-counting formula should be preferred?" The paper presents three and effectively says "doesn't matter." That's unsatisfying.

**What to do in the rebuttal text:**
- "We recommend the Standard Formula (Eqn. 1) for three reasons: (1) it is derived from architectural first principles with no free parameters, (2) it produces the flattest D/N ratio with compute (slope = -0.572 per decade vs -1.049 and -1.248), strengthening the Chinchilla finding, and (3) it aligns with the convention of excluding certain parameters from N when computing $C \approx 6ND$, consistent with Kaplan et al."

**What to do in the revised manuscript:**
- Add this recommendation to the Discussion section as a concrete practical takeaway.

**Addresses:** EBqg (Q2, practical guidance).

### Priority 7: Add explicit limitations section and address code validation (targets rRfE, CheF)

rRfE wants "a clearer limitations section" — or alternatively, "the authors' commentary on why their work does not suffer from sufficient limitations" (they offer both options). CheF notes the paper lacks a dedicated limitations discussion. rRfE also wants confirmation that we validated the Epoch AI code implementation.

**What to do in the rebuttal text:**
- "We will add a Limitations section covering: (1) Our analysis is conditioned on the Chinchilla dataset and fitting methodology — we do not claim extrapolation to modern overtrained regimes; (2) We analyze only parameter-count perturbations, not data-side or optimizer-side sources of error; (3) Our perturbation analysis uses synthetically perturbed covariates, not ground-truth alternative measurements."
- On code validation: "Our code is adapted from Epoch AI's open-source implementation. We verified correctness by reproducing their published fit parameters. Additionally, our analytical predictions (Appendix C) independently confirm the empirical trends, providing a cross-check that does not depend on any particular implementation."

**What to do in the revised manuscript:**
- Add a Limitations paragraph to the Discussion.
- Add a sentence in Section 2 explicitly stating we reproduced Epoch AI's published results.

**Addresses:** rRfE (limitations — their ask; code validation — their soundness concern), CheF (limitations).

### Priority 8: Sharpen differentiation from Porian et al. and Pearce & Song (targets rRfE)

rRfE — and only rRfE — explicitly asks how this work "complements" Porian et al. and Pearce & Song. This is a request for clarification, not a challenge: rRfE already says the paper "makes a clear case for its originality."

**What to do in the rebuttal text:**
- "Those works each identified *specific* accounting decisions (head parameters, embeddings, optimizer settings) that explain discrepancies *between* Chinchilla and Kaplan. Our work is complementary but distinct: (1) we discover a previously unnoticed ambiguity *within* Chinchilla's own reported data, (2) we provide a *systematic* sensitivity analysis across four perturbation families rather than a single fix, and (3) we derive analytical explanations that yield a general diagnostic framework rather than case-by-case corrections. Indeed, our additive-constant analysis (Sec 3.2) *subsumes and explains* both Porian et al.'s and Pearce & Song's findings as special cases (as noted on manuscript page 6, where the quantitative agreement is shown)."

**Addresses:** rRfE (originality clarification).

### Priority 9: Address remaining reviewer-specific questions

**rRfE: "the extent to which Chinchilla continues to inform consequential training runs"**
- This is a distinct ask from "modern relevance" — rRfE wants concrete evidence that practitioners still rely on Chinchilla-style scaling. "Chinchilla's compute-optimal prescriptions directly informed the training of LLaMA 1 (Touvron et al., 2023), and its tokens-per-parameter heuristic remains a standard reference point in scaling discussions (e.g., DeepSeek, Llama 3). Even in overtrained regimes that deviate from the Chinchilla-optimal point, practitioners use the Chinchilla baseline to quantify *how much* they are overtraining."

**rRfE: Why were counts ambiguous?**
- "We suspect the discrepancy arose because the original Chinchilla paper may have used a non-standard attention parameter formula internally (our 'best fit' formula with a factor of 5 instead of 4, which could reflect including output projections or bias terms) while the table's architectural hyperparameters invite the 'standard' formula. This kind of silent definitional mismatch is common in large-scale ML projects. We will add a recommendation that future scaling papers fully specify their parameter-counting conventions."

**rRfE: Modern relevance (beyond the "continues to inform" question)**
- "While we scope our empirical claims to the Chinchilla regime, our *diagnostic framework* is regime-independent. The mathematical relationships between perturbation type and scaling parameter distortion hold for any power-law analysis. Practitioners in overtrained regimes can apply the same diagnostic logic to their own fits."

**CheF Q1 (additional data):**
- "To our knowledge, no other paper has published per-model pretraining losses at the granularity of Chinchilla's Table A9. Scaling law papers (e.g., LLaMA, DeepSeek) report aggregate trends but not the per-configuration data needed to re-fit the law. We agree that extending to additional datasets is a valuable future direction and will note this."

**CheF Q2 (changing functional form):**
- "We deliberately held the functional form fixed to isolate the effect of parameter-counting ambiguity. Changing the form simultaneously would confound the analysis. That said, our diagnostic framework reveals when an *apparent* change in functional form is actually an artifact of additive specification errors — precisely the guidance that helps the community avoid premature modifications to the law."

**CheF Q3 (theory-simulation consistency):**
- "Our analytical predictions and simulation results are consistent. For multiplicative constants, we prove $\hat{\alpha} = \alpha$ exactly (Appendix C.2.1). For systematic bias, we derive $\hat{\alpha} = \alpha/s$ and empirically observe $R^2 > 0.999$ (Sec. 3.3). For additive constants, we derive the direction and monotonicity of $\hat{\alpha}$'s shift, matching the empirical trend. Divergence occurs only at extreme magnitudes where the fitter encounters numerical issues (NaNs)."

**CheF: Data quality and diversity**
- CheF notes that our Related Work mentions data quality and diversity as relevant to scaling, but the analysis does not consider these factors. "We agree that data quality and diversity are important dimensions of scaling law robustness. Our paper focuses specifically on covariate specification errors (parameter counting) as a controlled, isolable source of uncertainty. Extending the sensitivity analysis to data-side perturbations (e.g., data quality filtering thresholds, domain composition) is a natural next step that our perturbation framework could accommodate, and we will note this in the revised manuscript."

**CheF: General novelty concern**
- CheF characterizes the contribution as "a new fit to the data in Hoffmann et al. (2022)." In the rebuttal: "We respectfully clarify that the paper is not a new fit to the Chinchilla data — it is a systematic investigation of how *wrong* the input covariates can be before the fit breaks. The three parameter counting formulas are not alternative scaling laws; they are three interpretations of the same model architectures. The perturbation analysis then generalizes beyond any specific formula to characterize the sensitivity of power-law fits to covariate misspecification, yielding analytical results (Appendix C) that apply broadly."
