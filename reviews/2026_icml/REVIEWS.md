# ICML 2026 Reviews — Submission 19414

**Title:** Miscounted Parameters, Misread Scaling: A Robustness Study of Chinchilla Compute-Optimal Scaling

**Authors:** Rylan Schaeffer, Noam Itzhak Levi, Andreas Kirsch, Theo Guenais, Brando Miranda, Elyas Obbad, Sanmi Koyejo

**Venue:** ICML 2026 Conference Submission

---

## Reviewer EBqg — Weak Reject (3)

**Confidence:** 2

### Summary

This paper claims that the parameter count used in the original Chinchilla scaling law paper is somewhat ambiguous. The authors investigate three model parameter-counting formulas for LLMs and find that the resulting counts can vary by up to 15.2%. However, all of the parameter-counting formulas still support the conclusions of the Chinchilla scaling law, indicating that it is quite robust. The authors also conduct additional studies on the robustness of Chinchilla's scaling law by perturbing the parameter counts. These experiments further demonstrate the robustness of the Chinchilla scaling law.

### Strengths

- The paper is generally well written and well motivated.
- Investigating the robustness of scaling laws for large language models is a meaningful and potentially impactful research direction.
- The observation regarding differences in parameter-counting paradigms is interesting.

### Weaknesses

- The paper primarily justifies the robustness of the Chinchilla scaling law, which has already been empirically validated many times in practice (e.g., it was adopted in training earlier series of LLaMA models). Therefore, it is not immediately clear why this contribution warrants a new paper.
- Based on the conclusions of the paper, it is still unclear which parameter-counting formula should be preferred over the others.
- The paper does not propose either a new (or modified) scaling law or provide new guidance on applying the Chinchilla scaling law (e.g., the commonly cited rule of 20 tokens per parameter), which limits the practical impact of the work.

### Key Questions

1. Given that the robustness of the Chinchilla scaling law has already been verified by many model developers, what additional contribution does this paper provide to the community?
2. Based on the empirical observations presented in the paper, which parameter-counting formula should be preferred over the others?
3. Do the findings of this paper suggest a new (or modified) scaling law, or provide new guidance on applying the Chinchilla scaling law (e.g., the commonly cited rule of 20 tokens per parameter)?

### Limitations

As mentioned earlier, I believe the major limitation of this paper is that it is purely an empirical study and does not provide practical guidance for LLM development.

**Soundness:** 3 (good) | **Presentation:** 3 (good) | **Significance:** 2 (fair) | **Originality:** 3 (good)

---

## Reviewer CheF — Weak Reject (3)

**Confidence:** 3

### Summary

The authors try to adjust the Chinchilla scaling laws and assert that they still hold under considerations of random noise and bias. The paper also perturbs existing data to test its robustness through simulation and theoretical analysis.

### Strengths

- The authors provide a detailed discussion of the Chinchilla scaling laws and demonstrate that the revised formula performs better and may be more generalizable.
- The authors used multiple perturbation experiments to verify the stability, showing that the Chinchilla scaling laws hold in at least these scenarios.

### Weaknesses

- The main contribution of this paper can be seen as a new fit to the data in Hoffmann et al. (2022), revising the original law's function and performing robustness tests.
- While this study has practical significance, the authors have not considered some practical issues, such as data quality and diversity mentioned in their related work.
- Given the data size and the form of the equation, a more in-depth theoretical analysis could be conducted.

### Key Questions

1. Apart from the data in Hoffmann et al. (2022), have any subsequent papers provided additional data points? If so, can the authors also validate them?
2. Have the authors considered making more substantial modifications to the Chinchilla scaling laws, such as changing the form of equation?
3. Is there any inconsistency between the theoretical analysis of the perturbations and the simulation results?

### Limitations

The paper does not include a dedicated discussion of its limitations, although some aspects are mentioned in sections such as related work and future research directions.

**Soundness:** 3 (good) | **Presentation:** 3 (good) | **Significance:** 2 (fair) | **Originality:** 2 (fair)

---

## Reviewer rRfE — Weak Accept (4)

**Confidence:** 2

### Summary

This paper scrutinizes the Chinchilla Scaling Laws [1] and finds that the exact model parameters detailed in that work were ambiguous. The authors posit that this ambiguity warrants a targeted study given the foundational nature of Chinchilla laws. Indeed, scaling laws like Chinchilla have helped motivate the field's efforts to scale up data and model sizes across successive training runs. Any potential ambiguities or flaws in the original analysis could have a broad effect on the field. The authors investigate this ambiguity by studying three methods for estimating the number of parameters in the Chinchilla scaling suite. They broadly find that the core scaling analysis withstands these modifications to the parameter counting process. Taken together, these results provide further evidence of the Chinchilla Scaling Law's robustness under trivial experimental conditions.

[1] - Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D.D., Hendricks, L.A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., Driessche, G.V., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J.W., Vinyals, O., & Sifre, L. (2022). Training Compute-Optimal Large Language Models. ArXiv, abs/2203.15556.

### Strengths

- **Soundness:** This work appears technically sound, with the parameter-counting and perturbation methods intuitive and, in part, motivated by concrete considerations. The main concern is that they appear to be relying on an external code implementation. Have the authors taken concrete steps to validate this implementation?
- **Presentation:** The paper is well-formatted overall, with no glaring issues that inhibit my ability to glean the findings. I would have liked to see a clearer limitations section. Alternatively, the authors' commentary on why their work does not suffer from sufficient limitations.
- **Significance:** Much of the progress in language modeling has come from scaling the FLOPS over successive training runs. Scaling laws like Chinchilla help inform the experimental considerations for these runs. These resource-intensive training runs leave little room for hyperparameter optimization and iteration. Increasing confidence in scaling laws is essential. This work addresses a concerning ambiguity within Hoffman et al. While the scope of this work is narrow, surfacing ambiguities in underlying scaling laws is so important. However, I'd be interested in the authors highlighting the extent to which Chinchilla continues to inform consequential training runs. Indeed, the authors claim in the abstract that "Our findings offer renewed confidence in Chinchilla as a durable guide for scaling modern models" while noting the non-modern scope, "We do not test whether Chinchilla's functional form extrapolates to modern training regimes; our claims are conditional on the Chinchilla dataset and fitting methods". A more nuanced future work section could help clarify the sort of research enabled by this paper.
- **Originality:** This work makes a clear case for its originality, while not claiming to be the only work to challenge the issues raised in the original Chinchilla paper. However, as the paper notes, other works have examined the role of design choices in determining the parameters that affect scaling-law results [1, 2]. I'd be excited for the authors to clarify how their work complements these existing works to better establish originality.

[1] - Porian, T., Wortsman, M., Jitsev, J., Schmidt, L., & Carmon, Y. (2024). Resolving Discrepancies in Compute-Optimal Scaling of Language Models. ArXiv, abs/2406.19146.
[2] - Pearce, T., & Song, J. (2024). Reconciling Kaplan and Chinchilla Scaling Laws. Trans. Mach. Learn. Res., 2024.

### Key Questions

Are the authors in a position to conjecture as to why the parameter counts in the Chinchilla paper were ambiguous in the first place? Ideally, works like this one should be unnecessary when sufficient model details are shared publicly. How can future scaling law papers avoid this in the future?

### Limitations

- Are the authors able to make a case for the significance of these results in modern training regimes that are vastly overtrained by Chinchilla standards?
- My understanding is that the authors are relying on an open-source replication of the original paper. I'd be excited for the authors to note this potential limitation, or clarify that they have taken steps to verify the validity of this implementation.

**Soundness:** 3 (good) | **Presentation:** 3 (good) | **Significance:** 2 (fair) | **Originality:** 3 (good)

---

## Reviewer v5yx — Weak Reject (3)

**Confidence:** 3

### Summary

This paper investigates the robustness of the compute-optimal scaling laws introduced by Hoffmann et al. (the "Chinchilla" paper). Specifically, the authors identify an ambiguity in how model parameters were counted. They evaluate how three different interpretations of these parameter counts affect key outcomes, namely the estimated scaling law parameters and the optimal token-to-parameter ratio. Furthermore, the authors conduct a sensitivity analysis by applying four types of structured distortions (multiplicative, additive, systematic bias, and log-normal noise) to the parameter counts. The paper concludes that the core Chinchilla heuristic (the ~20:1 ratio) does not meaningfully change across realistic interpretations.

### Strengths

- The identification of the ambiguity regarding model parameter counts is a valuable observation. Given how foundational the Chinchilla paper is to current LLM training paradigms, pointing out and rigorously testing this discrepancy is a solid contribution to the community.
- The empirical approach is well-structured. By evaluating both realistic alternative counting methods (Section 2) and conducting a structured sensitivity analysis with hypothetical perturbations (Section 3), the paper provides a comprehensive examination of how parameter scaling estimates respond to different types of noise and biases.

### Weaknesses

- In the introduction, the authors motivate the paper by citing "recent concerns regarding wide confidence intervals and inconsistent results," claiming this work will offer the field "renewed confidence." However, the paper does not adequately bridge the gap between these specific concerns and the proposed analysis. It remains unclear whether the authors are suggesting that the parameter count ambiguity is the root cause of the wide confidence intervals or the inconsistencies noted by prior work. If the parameter ambiguity is unrelated to those specific concerns, the introductory framing might be over-promising the scope of the resolution this paper provides.

### Key Questions

1. **Clarification on the Conclusion of "Robustness" (Regarding Figure 5):** In the text, the authors state that "overall, all four sensitivity analyses demonstrate that Chinchilla's key results withstand sizable perturbations." However, a direct visual inspection of Figure 5 seems to contradict this statement. The plots clearly show that the optimal token-to-parameter ratio deviates significantly and trends away from the D/N = 20 baseline under various perturbation scales (e.g., the curves fan out drastically). Could the authors clarify the criteria used to define "withstand" and "sizable"? It appears that while the ratio is robust within the realistic error margin (e.g., the ~15.2% discrepancy found in Section 2), it clearly breaks down under the more extreme stress tests shown in Figure 5. It would greatly improve the clarity of the paper if the authors explicitly discussed the threshold at which the heuristic fails, rather than making a generalized claim that might confuse readers looking at the divergent curves in Figure 5.

### Limitations

The primary limitations of this work are closely reflected in the points raised in the Weaknesses and Questions sections above. Please refer to those sections for a detailed discussion.

**Soundness:** 2 (fair) | **Presentation:** 1 (poor) | **Significance:** 2 (fair) | **Originality:** 2 (fair)
