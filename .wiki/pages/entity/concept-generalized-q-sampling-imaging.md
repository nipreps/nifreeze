---
title: "Generalized q-Sampling Imaging (GQI)"
entity_type: concept
namespace: paper
last_verified: 2026-07-12
confidence_score: 0.9
---

# Generalized q-Sampling Imaging (GQI)

A model-free q-space reconstruction method that estimates a **spin distribution
function (SDF)** directly from diffusion-weighted MR signals, derived from the
Fourier-transform relation between the MR signal and the underlying diffusion
displacement ([[yeh-2010-generalized-q-sampling-imaging]]). Unlike q-ball
imaging (QBI) and diffusion spectrum imaging (DSI), GQI applies to **any**
sampling scheme — shell (as in QBI) or grid (as in DSI) — because its
reconstruction is a single quadrature over the acquired q-space samples, with no
Fourier transform or grid interpolation step.

This page is domain theory (the mathematics of the technique at the fidelity of
the original paper); it does not describe any nifreeze code. See
[[concept-diffusion-mri-signal]] for the acquisition physics the symbols below
inherit ($\mathbf{q}$, $b$, $\rho$, antipodal symmetry).

## The spin distribution function (SDF)

QBI and DSI both reconstruct the **orientation distribution function (ODF)** — a
*probability* distribution of diffusion displacement that is normalized
independently in every voxel, so an ODF value in one voxel is not comparable to
the same value in another. GQI instead reconstructs the SDF, which represents
the *density of spins* undergoing diffusion in a given direction. Because the
SDF carries the spin density (not a per-voxel-normalized probability), SDF
values share a unified reference and **can be compared across voxels**. This
inter-voxel comparability is the property that later enables quantitative
anisotropy (QA, Eq. 11) and its correlation with fiber volume fraction.

## Derivation

### Fourier-transform foundation

Combined $k$-space and $q$-space imaging rests on the Fourier relation between
the MR signal $S(\mathbf{k}, \mathbf{q})$, the spin density $\rho(\mathbf{r})$,
and the average propagator $p_\Delta(\mathbf{r}, \mathbf{R})$ over diffusion time
$\Delta$:

$$
S(\mathbf{k}, \mathbf{q}) =
\int \rho(\mathbf{r})\exp(i 2\pi \mathbf{k}\cdot\mathbf{r})
\int p_\Delta(\mathbf{r}, \mathbf{R})\exp(i 2\pi \mathbf{q}\cdot\mathbf{R})
\, d\mathbf{R}\, d\mathbf{r}
\tag{1}
$$

where $\mathbf{r}$ is the voxel coordinate, $\mathbf{R}$ is the diffusion
displacement, $\mathbf{q} = \gamma \mathbf{G}\delta / 2\pi$ (with $\gamma$ the
gyromagnetic ratio, $\mathbf{G}$ the gradient strength and $\delta$ its
duration). $k$-space reconstruction yields the diffusion-weighted image data
$W(\mathbf{r}, \mathbf{q})$, which reveals each voxel's average propagator:

$$
W(\mathbf{r}, \mathbf{q}) =
\int \rho(\mathbf{r})\, p_\Delta(\mathbf{r}, \mathbf{R})
\exp(i 2\pi \mathbf{q}\cdot\mathbf{R})\, d\mathbf{R}
\tag{2}
$$

### Spin density function and the cosine transform

To express the propagator in units of spin *quantity*, define the spin density
function $Q(\mathbf{r}, \mathbf{R}) = \rho(\mathbf{r})\, p_\Delta(\mathbf{r},
\mathbf{R})$ — the propagator scaled by the density. Because $Q$ is real and
symmetric in $q$-space, $W(\mathbf{r}, \mathbf{q}) = W(\mathbf{r}, -\mathbf{q})$,
so $Q$ is recovered by a **cosine** transform (no imaginary part survives):

$$
Q(\mathbf{r}, \mathbf{R}) =
\int W(\mathbf{r}, \mathbf{q})\cos(2\pi \mathbf{q}\cdot\mathbf{R})\, d\mathbf{q}
\tag{3}
$$

### Integrating spins along a direction → the SDF

The quantity of spins diffusing in a particular direction $\hat{\mathbf{u}}$ is
obtained by integrating $Q$ along the ray $\mathbf{R} = L\hat{\mathbf{u}}$ up to
a **diffusion sampling length** $L_\Delta$:

$$
\psi_Q(\mathbf{r}, \hat{\mathbf{u}}) =
\int_0^{L_\Delta} Q(\mathbf{r}, L\hat{\mathbf{u}})\, dL
\tag{4}
$$

Equation (4) shows the SDF is an orientation distribution function *of the spin
quantity* (it equals the diffusion ODF multiplied by the spin density).

### The GQI reconstruction equation

Substituting (3) into (4) and evaluating the inner integral over $L \in [0,
L_\Delta]$ turns the cosine into a $\operatorname{sinc}$:

$$
\psi_Q(\mathbf{r}, \hat{\mathbf{u}}) =
\int_0^{L_\Delta}\!\!\int W(\mathbf{r}, \mathbf{q})
\cos(2\pi L\,\mathbf{q}\cdot\hat{\mathbf{u}})\, d\mathbf{q}\, dL
= L_\Delta \int W(\mathbf{r}, \mathbf{q})
\operatorname{sinc}(2\pi L_\Delta\,\mathbf{q}\cdot\hat{\mathbf{u}})\, d\mathbf{q}
\tag{5}
$$

where $\operatorname{sinc}(x) = \sin(x)/x$ for $x \neq 0$ and
$\operatorname{sinc}(0) = 1$. Equation (5) exhibits the SDF as a superposition of
**basis SDFs** weighted by $W(\mathbf{r}, \mathbf{q})$; the shape of each basis
SDF is set by $|\mathbf{q}| L_\Delta$ — a larger value gives a sharper contour.

Replacing the integral with a quadrature over the discrete acquired samples
gives the **measured SDF** — the operative GQI reconstruction formula,
applicable to any sampling scheme:

$$
\psi_m(\mathbf{r}, \hat{\mathbf{u}}) =
A_q L_\Delta \sum_{\mathbf{q}} W(\mathbf{r}, \mathbf{q})
\operatorname{sinc}(2\pi L_\Delta\,\mathbf{q}\cdot\hat{\mathbf{u}})
\tag{6}
$$

where $A_q$ is a constant area term for the quadrature. The SDF may additionally
be scaled by a constant $Z_0$ so that the SDF of pure water is 1; in practice
cerebrospinal fluid (free diffusion) is used as the $Z_0$ reference. **This SDF
scaling differs fundamentally from the ODF normalization in QBI/DSI**: ODF
normalization is per-voxel (to make each a probability density), whereas the SDF
scaling is one constant applied to all voxels, preserving inter-voxel
comparability. (Normalizing the SDF instead per-voxel would turn it back into a
diffusion ODF.)

## Relation to other q-space methods

- **QBI is a special case of GQI.** If the diffusion sampling length
  $L_\Delta \to \infty$ in Eq. (5), the $\operatorname{sinc}$ approaches a delta
  function and the reconstruction reduces to the **Funk–Radon transform** used by
  QBI ([[tuch-2004-q-ball-imaging]] via [[yeh-2010-generalized-q-sampling-imaging]]).
  GQI is thus the general finite-$L_\Delta$ approach of which QBI is the infinite
  limit.
- **DSI shares the same theoretical basis** (the Fourier relation of Eq. 1) but
  differs numerically: DSI applies a Fourier transform to grid $q$-space data,
  then numerically integrates the ODF on the transformed grid — and regularizes
  with a Hanning filter to suppress truncation artifacts. GQI performs a direct
  mathematical reduction (Fourier + ODF integral collapsed into one quadrature),
  avoiding the Fourier transform, the grid interpolation, and the Hanning
  filter; it uses the explicit control parameter $L_\Delta$ to trade angular
  resolution against artifact instead.
- **DSI's ODF definition carries an extra $L^2$ (solid-angle) weighting**, a
  Jacobian from transforming the propagator into an ODF:

  $$
  \psi(\hat{\mathbf{u}}) =
  \int_0^{\infty} p_\Delta(L\hat{\mathbf{u}})\, L^2\, dL
  \tag{7}
  $$

  Carrying this $L^2$ weighting through the GQI reduction yields a different
  basis function $f$:

  $$
  \psi_m(\mathbf{r}, \hat{\mathbf{u}}) =
  A_q L_\Delta^3 \sum_{\mathbf{q}} W(\mathbf{r}, \mathbf{q})\,
  f\!\left(L_\Delta\,\mathbf{q}\cdot\hat{\mathbf{u}}\right)
  \tag{8}
  $$

  $$
  f(x) =
  \begin{cases}
  \dfrac{2\cos(x)}{x^2} + \dfrac{(x^2 - 2)\sin(x)}{x^3}, & x \neq 0 \\[2mm]
  \dfrac{1}{3}, & x = 0
  \end{cases}
  $$

  The paper notes there is no guarantee which $L$-weighting gives better
  orientation estimation; the study used the $\operatorname{sinc}$ basis (Eq. 6),
  not the $L^2$ basis (Eq. 8), for reconstruction.

## The diffusion sampling length $L_\Delta$

$L_\Delta$ (Eq. 4) sets the range of diffusion displacement integrated, and so
acts as a **regularization / smoothing parameter**: a lower $L_\Delta$ covers
less displacement (coarser SDFs), a higher $L_\Delta$ covers more (sharper
SDFs). Under Gaussian diffusion the natural scale is the diffusion length
$(6 D \tau)^{1/2}$, where $D$ is the diffusion coefficient and

$$
\tau = \Delta - \tfrac{\delta}{3}
$$

is the effective diffusion time. $L_\Delta$ is expressed as a multiple of this
length,

$$
L_\Delta = \sigma\,(6 D \tau)^{1/2},
$$

with $\sigma$ an adjustable factor. At $\sigma = 1.25$ roughly 80% of the
diffusion distribution is encompassed (higher under restricted diffusion). The
paper reports $\sigma$ between **1 and 1.3** gives good reconstructions;
$\sigma > 1.3$ raises noise sensitivity.

### $b$-value form of the reconstruction

Because the $b$-value is more commonly reported than $q$, substituting
$L_\Delta = \sigma (6 D \tau)^{1/2}$ into Eq. (6) recasts the reconstruction in
terms of $b$ and $\sigma$:

$$
\psi_m(\mathbf{r}, \hat{\mathbf{u}}) =
A_q L_\Delta \sum_{\mathbf{q}} W(\mathbf{r}, \mathbf{q})
\operatorname{sinc}\!\left(
\sigma \sqrt{6 D \cdot b(\mathbf{q})}\;
\frac{\mathbf{q}}{|\mathbf{q}|}\cdot\hat{\mathbf{u}}
\right)
\tag{9}
$$

where $b(\mathbf{q}) = (2\pi|\mathbf{q}|)^2\,\tau$ is the $b$-value of the
encoding, and $\mathbf{q}/|\mathbf{q}|$ is the unit gradient direction. This is
the practical form: it takes $b$-values and $\sigma$ as input instead of $q$.

## Applicable sampling schemes: the balanced requirement

GQI applies to any scheme, but the result is only correct if the scheme is
**balanced**: MR signals from *isotropic* diffusion must reconstruct to an
*isotropic* SDF. A quick numerical test assumes signals generated by an
isotropic tensor $D_I$ and computes the SDF that should result:

$$
\psi_0(\hat{\mathbf{u}}) =
A_q L_\Delta \sum_{\mathbf{q}}
\operatorname{sinc}(2\pi L_\Delta\,\mathbf{q}\cdot\hat{\mathbf{u}})\,
\exp(-4\pi^2\,\mathbf{q}^{\mathsf{T}} D_I\,\mathbf{q}\,\tau)
\tag{10}
$$

A scheme is acceptable if $\psi_0(\hat{\mathbf{u}})$ is nearly isotropic (low
variance across $\hat{\mathbf{u}}$). The balanced requirement is a *necessary*
condition for correct reconstruction (sufficiency is left open) and provides a
framework for designing GQI-optimized clinical sampling schemes — including
subsampling an existing shell/grid scheme down to fewer directions while staying
balanced.

## Quantitative anisotropy (QA)

QA quantifies the spin population along a *specific resolved fiber orientation*
$\hat{\mathbf{a}}$ (a maximum of the SDF), unlike FA or GFA which are single
per-voxel scalars. QA subtracts the background isotropic component from the SDF
value at $\hat{\mathbf{a}}$:

$$
QA(\hat{\mathbf{a}}) =
Z_0\big(\psi_Q(\hat{\mathbf{a}}) - I(\psi_Q)\big)
\tag{11}
$$

where $Z_0$ is the SDF scaling constant and $I(\psi_Q)$ is the isotropic
background component — approximated in the paper by the minimum value of
$\psi_Q$. Because QA is per-fiber and inter-voxel-comparable, it can be compared
against fiber-specific quantities such as volume fraction. The simulation found
QA correlated with fiber volume fraction ($r = 0.86$, $p < 0.01$), negatively
with the isotropic-background fraction ($r = -0.33$), and modestly with FA
($r = 0.38$).

## Validation status

Evaluated by a mixed-Gaussian simulation (two fiber populations plus an
isotropic component) and an in-vivo experiment, compared against QBI (shell
data) and DSI (grid data). The simulation signal model was

$$
S(b, \mathbf{v}) = S(0)\big(
f_1 \exp(-b\,\mathbf{v}^{\mathsf{T}} D_1 \mathbf{v}) +
f_2 \exp(-b\,\mathbf{v}^{\mathsf{T}} D_2 \mathbf{v}) +
f_0 \exp(-b\,\mathbf{v}^{\mathsf{T}} D_0 \mathbf{v})
\big)
\tag{12}
$$

with $f_1, f_2$ the fiber volume fractions, $f_0$ the isotropic fraction, and
$D_0, D_1, D_2$ the corresponding tensors; Rician noise at b0-SNR $= 30$.

Findings: GQI's angular deviation of the major fiber and its success rate at
resolving minor (crossing) fibers were **comparable to QBI on shell data and to
DSI on grid data** — the paper explicitly does *not* claim GQI has better angular
resolution. Accuracy depends on $L_\Delta$. In-vivo SDF patterns and tractography
were similar to those from QBI/DSI.

**Acknowledged limitations.** (1) A correct SDF is not guaranteed unless the
scheme is balanced. (2) The measured density $\rho(\mathbf{r})$ is affected by
$T_1$, $T_2$, and B1 inhomogeneity (mitigable via longer TR, an extra b0 at
different TE, and improved RF designs). (3) The SDF pattern changes with the
sampling scheme, so an optimum scheme is needed.

## Relevance to nifreeze

GQI is a reconstruction/analysis technique in the same q-space family as the
signal models nifreeze fits and predicts (see
[[concept-diffusion-mri-signal]]); it is not itself a motion/distortion
correction step. Its load-bearing idea for this KB is the SDF's **inter-voxel
comparability** — a design contrast against the per-voxel-normalized ODF — and
the use of an explicit smoothing length $L_\Delta$ as a tunable regularizer
rather than a post-hoc filter. See [[q-space-reconstruction-landscape]] for how
GQI sits relative to QBI and DSI.
