---
title: "Jensen et al. (2005) — Diffusional kurtosis imaging: quantification of non-Gaussian water diffusion by MRI"
entity_type: paper
doi: 10.1002/mrm.20508
s2_id: 62f5553e3db964055b36ece9f3ed319d7e038191
citation_count_at_verify: 2424
source_urls:
  - https://doi.org/10.1002/mrm.20508
last_verified: 2026-07-13
confidence_score: 0.85
refresh_needed_if: "DOI changes or paper retracted"
---

# Jensen et al. (2005) — Diffusional kurtosis imaging (DKI)

Jensen, J. H., Helpern, J. A., Ramani, A., Lu, H. and Kaczynski, K. (2005).
*Diffusional kurtosis imaging: the quantification of non-Gaussian water
diffusion by means of magnetic resonance imaging.* Magnetic Resonance in
Medicine **53**(6):1432–1440.

## Why relevant

**Claim.** Water diffusion in biological tissue is **not** Gaussian: at higher
b-values the log-signal deviates from the straight line the diffusion tensor
predicts. DKI extends DTI by adding the next term of the cumulant expansion of
the diffusion-weighted signal — a fourth-order **kurtosis tensor** capturing the
leading non-Gaussian deviation — which quantifies microstructural complexity the
tensor alone cannot.

**Evidence.** The paper derives the signal model
$\ln S(b) = \ln S_0 - b\,D_{\mathrm{app}} + \tfrac{1}{6}b^2 D_{\mathrm{app}}^2
K_{\mathrm{app}} + \mathcal{O}(b^3)$, where $K_{\mathrm{app}}$ is the apparent
kurtosis, and shows in vivo that the extra term is measurable and reproducible.
Because the fit resolves two parameters per direction ($D$ and $K$), it requires
sampling **at least three b-value levels** (including $b=0$), unlike the two the
tensor needs.

**Location in the project.** This is the theory backing
[[concept-diffusion-kurtosis-imaging]], which grounds DIPY's
`DiffusionKurtosisModel` (wrapped by nifreeze in
`src/nifreeze/model/dki.py`). The ≥3-shell requirement is why nifreeze gates the
DKI model on `check_multi_b(gtab, 3, non_zero=False)` (see
[[tool-dipy-gradient-table]] and [[tool-dipy-reconst-models]]).

**Limits.** The cumulant expansion is valid only over a limited b-range
(roughly $b \lesssim 3000\,\mathrm{s/mm^2}$); at very high b it breaks down.
Kurtosis estimation is more sensitive to noise than tensor estimation, motivating
the robust/weighted fitting variants DIPY exposes.
