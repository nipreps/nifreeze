---
title: "Rigid-body head motion"
entity_type: concept
namespace: paper
last_verified: 2026-07-08
confidence_score: 0.9
---

# Rigid-body head motion

The 6-degree-of-freedom model of subject movement between volumes — the "motion"
half of motion-and-distortion correction. Notation follows
[[andersson-2016-integrated-eddy]]; the geometry is standard computer-vision
rigid registration.

## The transform

Head motion between the reference and volume $i$ is a **rigid body** transform
$r_i$: three translations and three rotations, represented by a rotation matrix
$R_i$ and a translation $t$. A rotation about the $x$ axis, for example, is
(Eq. A2)

$$ R_x = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & \sin\theta \\ 0 & -\sin\theta & \cos\theta \end{bmatrix}, $$

with $R = R_x R_y R_z$ the full rotation. Mapping a reference-space voxel to
acquisition space, in homogeneous coordinates and accounting for the
voxel→world matrices $M_S$ (reference) and $M_Q$ (acquisition), is (Eq. A1)

$$ \begin{bmatrix} x' \\ 1 \end{bmatrix} = M_S^{-1} \begin{bmatrix} R & t \\ 0^{\mathsf{T}} & 1 \end{bmatrix}^{-1} M_Q \begin{bmatrix} x \\ 1 \end{bmatrix}. $$

## Coupling with the off-resonance fields

Motion does not act in isolation. The **order of operations matters** because
the two off-resonance fields transform differently under movement
([[concept-epi-off-resonance-distortion]], [[concept-eddy-current-distortion]]).
The composite mapping from reference voxel $x$ to the sampling location $x'$ in
the observed volume is (Eq. 2)

$$ x' = R_i^{-1} x + d_x\big(h + \omega(e(\beta_i), r_i),\, a_i\big), $$

where $\omega(\psi, r)$ rotates a field with the subject and $d_x(\cdot, a)$ is
the PE-direction displacement. The **susceptibility** field $h$ is fixed in
scanner space (not rotated), whereas the **eddy** field $e(\beta_i)$ *is* moved
with the subject via $\omega(\cdot, r_i)$ — the physically-correct combination.
(Eq. 3 gives the equivalent form with the alternative composition order.)

## Grounding in the project

Rigid-body motion is the transform nifreeze estimates per volume with ANTs, by
registering each observed volume to the model prediction
([[gp-prediction-underpins-lovo]], [[concept-image-registration]]). This page is
the theory of the transform being recovered and, critically, *why the fields it
is entangled with must be composed in the right frame* — a correctness
constraint for any reimplementation.
