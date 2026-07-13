---
title: "Sphere sampling: electrostatic repulsion and octahedral subdivision"
entity_type: concept
namespace: paper
last_verified: 2026-07-13
confidence_score: 0.9
---

# Sphere sampling: electrostatic repulsion and octahedral subdivision

Two distinct problems in diffusion MRI both reduce to *placing points on a
sphere*, but they have opposite goals and opposite methods. This page separates
them, because nifreeze uses both. It is domain theory; it does not describe
nifreeze code. Directions live on the half-sphere by antipodal symmetry
(see [[concept-diffusion-mri-signal]]).

## Problem 1 — choosing acquisition directions (electrostatic repulsion)

When designing a diffusion acquisition you must pick $N$ **diffusion-encoding
directions** that cover the sphere as uniformly as possible, so no orientation is
under-sampled. The classic solution treats each direction as a **point charge**
constrained to the unit sphere and minimises the electrostatic (Coulomb) potential
energy ([[jones-1999-optimal-strategies-measuring-diffusion-anisotropic-systems]]):

$$ E = \sum_{i<j}\left(\frac{1}{\lVert \mathbf{g}_i - \mathbf{g}_j\rVert}
 + \frac{1}{\lVert \mathbf{g}_i + \mathbf{g}_j\rVert}\right). $$

The second term includes the **antipode** $-\mathbf{g}_j$, because diffusion
signal is antipodally symmetric — repelling both $\mathbf{g}_j$ and $-\mathbf{g}_j$
keeps the *half-sphere* coverage uniform. Gradient descent on $E$ from a random
initialisation relaxes the points to a near-uniform, minimum-energy configuration.
The result is a small, optimised set (tens to hundreds of directions) tuned to the
acquisition budget.

Key properties: the configuration is **data-independent**, only *approximately*
uniform (no exact equidistribution exists for arbitrary $N$), and depends on the
random start (many local minima of comparable energy).

## Problem 2 — tessellating the sphere for reconstruction (octahedral subdivision)

Model-free reconstruction (ODF/SDF methods such as
[[concept-generalized-q-sampling-imaging]]) needs a **dense, fixed mesh** of
directions on which to evaluate and compare an orientation function. Here the goal
is a deterministic, reproducible, near-regular tessellation — not an optimised
small set. The standard construction starts from a regular polyhedron (an
**octahedron**) and **recursively subdivides** each triangular face, projecting
new vertices onto the unit sphere. Each subdivision roughly quadruples the face
count; a recursion level $\ell$ yields a sphere with

$$ N_{\text{vertices}} = 4^{\ell} + 2 $$

vertices (e.g. $\ell=5 \Rightarrow 1026$ vertices). Unlike repulsion, this is
deterministic and arbitrarily refinable, at the cost of not being an
energy-minimal point set.

## Why the distinction matters

The two methods are not interchangeable: repulsion gives a *small optimised
acquisition scheme*; subdivision gives a *dense deterministic reconstruction
grid*. Using a subdivided mesh as an acquisition scheme would waste scan time;
using a repulsion set as a reconstruction sphere would give an irregular,
non-reproducible mesh.

## Relation to nifreeze's dependency

DIPY implements electrostatic repulsion via `HemiSphere` + `disperse_charges`
(used in nifreeze's phantom simulation to build gradient tables) and octahedral
subdivision via `create_unit_sphere` (used as the default ODF sphere inside the
vendored GQI model). Both contracts are cached in [[tool-dipy-sphere-directions]].
The multi-tensor signal simulated *on* these directions is
[[concept-multi-tensor-signal-simulation]].
