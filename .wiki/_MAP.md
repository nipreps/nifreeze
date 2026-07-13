# Wiki Relationship Map

Edges between pages. Each row links a source page to a target page through
one of the allowed relationship types defined in [`schema.md`](schema.md):
`cites`, `critiques`, `supersedes`, `extends`, `depends_on`, `informs`,
`implements`, `operationalizes`, `replicates`, `contradicts`.

A page that is the target of any edge here is exempt from the orphan rule
(W002) even if it is not linked from `index.md`.

| source | type | target | note |
|--------|------|--------|------|
| refs/andersson-2016-integrated-eddy.md | extends | refs/andersson-2015-gp-dmri.md | registers to GP prediction |
| refs/andersson-2016-outlier-replacement.md | extends | refs/andersson-2016-integrated-eddy.md | adds outlier handling |
| refs/andersson-2016-outlier-replacement.md | depends_on | refs/andersson-2015-gp-dmri.md | reuses GP predictor |
| refs/andersson-2015-gp-dmri.md | informs | pages/entity/concept-gaussian-process-regression.md | theory grounding |
| refs/andersson-2015-gp-dmri.md | informs | pages/entity/concept-dmri-angular-covariance.md | theory grounding |
| refs/andersson-2015-gp-dmri.md | informs | pages/entity/concept-diffusion-mri-signal.md | theory grounding |
| refs/andersson-2016-integrated-eddy.md | informs | pages/entity/concept-image-registration.md | theory grounding |
| refs/andersson-2016-integrated-eddy.md | informs | pages/entity/concept-eddy-current-distortion.md | theory grounding |
| refs/andersson-2016-integrated-eddy.md | informs | pages/entity/concept-epi-off-resonance-distortion.md | theory grounding |
| refs/andersson-2016-integrated-eddy.md | informs | pages/entity/concept-rigid-body-motion.md | theory grounding |
| refs/andersson-2016-outlier-replacement.md | informs | pages/entity/concept-outlier-detection-replacement.md | theory grounding |
| pages/synthesis/gp-prediction-underpins-lovo.md | depends_on | refs/andersson-2015-gp-dmri.md | predictive mean = LOVO target |
| pages/synthesis/andersson-eddy-framework-lineage.md | depends_on | refs/andersson-2016-integrated-eddy.md | methodological template |
| refs/andersson-2015-gp-dmri.md | informs | pages/entity/concept-leave-one-volume-out.md | theory grounding |
| refs/andersson-2016-integrated-eddy.md | informs | pages/entity/concept-leave-one-volume-out.md | theory grounding |
| pages/synthesis/gp-prediction-underpins-lovo.md | depends_on | pages/entity/concept-leave-one-volume-out.md | independence invariant |
| pages/synthesis/single-fit-mode-admissibility.md | depends_on | pages/entity/concept-leave-one-volume-out.md | admissibility hinges on independence |
| pages/synthesis/single-fit-mode-admissibility.md | depends_on | refs/andersson-2015-gp-dmri.md | LOVO/predictor grounding |
| refs/andersson-2015-gp-dmri.md | cites | refs/minka-2017-learning-how-learn-learning-point-sets.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/setsompop-2013-pushing-limits-vivo-diffusion-mri-human.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/setsompop-2013-pushing-limits-vivo-diffusion-mri-human.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/setsompop-2013-pushing-limits-vivo-diffusion-mri-human.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/ugurbil-2013-pushing-spatial-temporal-resolution-functional-diffusion.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/ugurbil-2013-pushing-spatial-temporal-resolution-functional-diffusion.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/ugurbil-2013-pushing-spatial-temporal-resolution-functional-diffusion.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/sotiropoulos-2013-advances-diffusion-mri-acquisition-processing-human.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/sotiropoulos-2013-advances-diffusion-mri-acquisition-processing-human.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/sotiropoulos-2013-advances-diffusion-mri-acquisition-processing-human.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/essen-2013-wu-minn-human-connectome-project-overview.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/essen-2013-wu-minn-human-connectome-project-overview.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/essen-2013-wu-minn-human-connectome-project-overview.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/pannek-2012-homor-higher-order-outlier-rejection-high.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/pannek-2012-homor-higher-order-outlier-rejection-high.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/panagiotaki-2012-compartment-models-diffusion-mr-signal-brain.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/descoteaux-2011-multiple-q-shell-diffusion-propagator-imaging.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/huang-2011-validity-commonly-used-covariance-variogram-functions.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/andersson-2010-image-distortion-its-correction-diffusion-mri.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/andersson-2010-image-distortion-its-correction-diffusion-mri.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/andersson-2010-image-distortion-its-correction-diffusion-mri.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/aganj-2010-reconstruction-orientation-distribution-function-single-mul.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/rathi-2008-directional-functions-orientation-distribution-estimation.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/descoteaux-2006-apparent-diffusion-coefficients-high-angular-resolutio.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/descoteaux-2006-apparent-diffusion-coefficients-high-angular-resolutio.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/chang-2005-restore-robust-estimation-tensors-outlier-rejection.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/chang-2005-restore-robust-estimation-tensors-outlier-rejection.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/rohde-2004-comprehensive-approach-correction-motion-distortion-diffusi.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/rohde-2004-comprehensive-approach-correction-motion-distortion-diffusi.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/behrens-2003-characterization-propagation-uncertainty-diffusionweighte.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/behrens-2003-characterization-propagation-uncertainty-diffusionweighte.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/rasmussen-2003-gaussian-processes-machine-learning.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/andersson-2002-model-based-method-retrospective-correction-geometric-d.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/andersson-2002-model-based-method-retrospective-correction-geometric-d.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/genton-2002-classes-kernels-machine-learning-statistics-perspective.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/sundararajan-1999-predictive-approaches-choosing-hyperparameters-gauss.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/basser-1994-estimation-effective-self-diffusion-tensor-nmr-spin.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/dempster-1977-maximum-likelihood-incomplete-em-algorithm.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/andersson-2013-865-gaussian-process-based-method-detecting.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/andersson-2013-865-gaussian-process-based-method-detecting.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/anon-2007-numericalrecipies-theartofscienti-fi-c-computing-chapter.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/alexander-2006-hybrid-diffusion-imaging-hydi.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/alexander-2006-hybrid-diffusion-imaging-hydi.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/mackay-2004-information-theory-inference-learning-algorithms.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/wackernagle-1998-multivariate-geostatistics-introduction-applications.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/wackernagle-1998-multivariate-geostatistics-introduction-applications.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/kass-1995-bayes-factors.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/dempster-1972-max-imum-likelihood-incomplete.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/nelder-1965-simplex-method-function-minimization.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/wang-nd-neuroimage.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/wang-nd-neuroimage.md |  |
| refs/andersson-2015-gp-dmri.md | cites | refs/bio-nd-apparent-diffusion-coefficients-high-angular-resolution.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/andersson-2015-non-parametric-representation-prediction-single-multi-s.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/andersson-2015-non-parametric-representation-prediction-single-multi-s.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/price-2015-technical-note-characterization-correction-gradient-nonline.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/vu-2015-high-resolution-whole-brain-diffusion-imaging.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/vu-2015-high-resolution-whole-brain-diffusion-imaging.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/graham-2015-simulation-framework-quantitative-validation-artefact-corr.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/mohammadi-2015-high-resolution-diffusion-kurtosis-imaging-3t-enabled.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/chan-2014-characterization-correction-eddy-current-artifacts-unipolar-.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/filippini-2014-study-protocol-whitehall-ii-imaging-sub-study.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/filippini-2014-study-protocol-whitehall-ii-imaging-sub-study.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/xu-2013-prospective-retrospective-high-order-eddy-current.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/zhuang-2013-correction-eddy-current-distortions-high-angular.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/caruyer-2013-design-multishell-sampling-schemes-uniform-coverage.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/mcnab-2013-human-connectome-project-beyond-initial-applications.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/maclaren-2013-prospective-motion-correction-brain-imaging-review.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/sotiropoulos-2013-rubix-combining-spatial-resolutions-bayesian-inferen.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/mcnab-2012-surface-based-analysis-diffusion-orientation-identifying.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/baron-2012-effect-concomitant-gradient-fields-diffusion-tensor.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/calamante-2012-comment-time-varying-eddy-currents-effects-diffusion-we.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/sotiropoulos-2012-ball-rackets-inferring-fibre-fanning-diffusion-weigh.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/farber-2011-cuda-application-design-development.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/setsompop-2011-blipped-controlled-aliasing-parallel-imaging-blipped-ca.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/setsompop-2011-blipped-controlled-aliasing-parallel-imaging-blipped-ca.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/truong-2011-dynamic-correction-artifacts-due-susceptibility-effects.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/wilm-2011-higher-order-reconstruction-mri-presence-spatiotemporal.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/bihan-2010-magnetic-resonance-diffusion-imaging-introduction-concepts.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/embleton-2010-distortion-correction-diffusionweighted-mri-tractography.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/moeller-2010-multiband-multislice-ge-epi-7-tesla-16-fold.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/moeller-2010-multiband-multislice-ge-epi-7-tesla-16-fold.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/leemans-2009-bmatrix-must-be-rotated-when-correcting.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/finsterbusch-2009-eddycurrent-compensated-diffusion-weighting-single-r.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/assaf-2008-axcaliber-method-measuring-axon-diameter-distribution.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/katscher-2007-parallel-magnetic-resonance-imaging.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/storey-2007-partial-kspace-reconstruction-singleshot-diffusionweighted.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/storey-2007-partial-kspace-reconstruction-singleshot-diffusionweighted.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/zhuang-2006-correction-eddycurrent-distortions-diffusion-tensor-images.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/skare-2006-propeller-epi-other-direction.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/chen-2006-correction-direction-dependent-distortions-diffusion-tensor-.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/shen-2004-correction-highorder-eddy-current-induced-geometric.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/janke-2004-use-spherical-harmonic-deconvolution-methods-compensate.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/morgan-2004-correction-spatial-distortion-epi-due-inhomogeneous.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/anon-2004-gaussian-processes-machine-learning.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/anon-2004-gaussian-processes-machine-learning.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/bodammer-2004-eddy-current-correction-diffusionweighted-imaging-pairs.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/andersson-2003-how-correct-susceptibility-distortions-spin-echo-echo-p.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/reese-2003-reduction-eddycurrentinduced-distortion-diffusion-mri-twice.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/reese-2003-reduction-eddycurrentinduced-distortion-diffusion-mri-twice.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/bastin-2001-use-flair-technique-improve-correction-eddy.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/jenkinson-2001-global-optimisation-method-robust-affine-registration.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/bastin-2000-use-water-phantom-images-calibrate-correct.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/horsfield-1999-mapping-eddy-current-induced-fields-correction.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/bastin-1999-correction-eddy-current-induced-artefacts-diffusion-tensor.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/jones-1999-optimal-strategies-measuring-diffusion-anisotropic-systems.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/jones-1999-optimal-strategies-measuring-diffusion-anisotropic-systems.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/kim-1999-motion-correction-fmri-registration-individual-slices.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/kim-1999-motion-correction-fmri-registration-individual-slices.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/calamante-1999-correction-eddy-current-induced-bo-shifts.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/andersson-1998-how-obtain-high-accuracy-image-registration-application.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/jezzard-1998-characterization-correction-eddy-current-artifacts-echo.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/alexander-1997-elimination-eddy-current-artifacts-diffusionweighted-ec.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/haselgrove-1996-correction-distortion-echoplanar-images-used-calculate.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/jezzard-1995-correction-geometric-distortion-echo-planar-images.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/unser-1993-b-spline-signal-processing-ii-efficiency-design.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/unser-1993-b-spline-signal-processing-i-theory.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/anon-2011-chapter-18-artifacts-diffusion-mri.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/basser-2009-diffusion-mri-quantitative-measurement-vivo-neuroanatomy.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/anon-2008-openmp-application-program-interface-version-3.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/anon-2007-numerical-recipies-art-scientific-computing-chapter.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/smith-2004-advances-functional-structural-mr-image-analysis.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/smith-2004-advances-functional-structural-mr-image-analysis.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/schmitt-1998-echo-planar-imaging-theory-technique-application.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/unser-1993-b-spline-signal-processing-part-ii-efficient-design.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/unser-1993-b-spline-signal-processing-part-i-theory.md |  |
| refs/andersson-2016-integrated-eddy.md | cites | refs/andersson-nd-modelling-geometric-deformations-epi-time-series.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/andersson-nd-modelling-geometric-deformations-epi-time-series.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/andersson-2016-integrated-approach-correction-off-resonance-effects-su.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/graham-2016-realistic-simulation-artefacts-diffusion-mri-validating.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/krogsrud-2016-changes-white-matter-microstructure-developing-braina.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/raffelt-2015-connectivity-based-fixel-enhancement-whole-brain-statisti.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/walker-2015-diffusion-tensor-imaging-dti-component-nih.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/collier-2015-iterative-reweighted-linear-least-squares-accurate.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/tax-2015-rekindle-robust-extraction-kurtosis-indices-linear.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/oguz-2014-dtiprep-quality-control-diffusion-weighted-images.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/wibral-2013-local-active-information-storage-tool-understand.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/yendiki-2013-spurious-group-differences-due-head-motion.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/perea-2013-comparative-white-matter-study-parkinson-s.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/mohammadi-2012-retrospective-correction-physiological-noise-dti-extend.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/heidemann-2012-k-space-q-space-combining-ultra-high-spatial-angular.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/chang-2012-informed-restore-method-robust-estimation-diffusion.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/maddah-2011-sheet-like-white-matter-fiber-tracts-representation.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/zhou-2011-automated-artifact-detection-removal-improved-tensor.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/nagel-2011-altered-white-matter-microstructure-children-attention.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/pierpaoli-2010-artifacts-diffusion-mri.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/zwiers-2010-patching-cardiac-head-motion-artefacts-diffusion-weighted.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/drobnjak-2010-simulating-effects-time-varying-magnetic-fields-realisti.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/westlye-2010-life-span-changes-human-brain-white-matter.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/yushkevich-2007-structure-specific-statistical-mapping-white-matter-tr.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/drobnjak-2006-development-functional-magnetic-resonance-imaging-simula.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/smith-2006-tract-based-spatial-statistics-voxelwise-analysis-multi-sub.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/group-2006-nih-mri-study-normal-brain-development.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/nunes-2005-investigations-efficiency-cardiac-gated-methods-acquisition.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/jones-2004-squashing-peanuts-smashing-pumpkins-how-noise.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/jones-2003-determining-visualizing-uncertainty-estimates-fiber-orienta.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/mangin-2002-distortion-correction-robust-tensor-estimation-mr.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/skare-2001-effects-gating-diffusion-imaging-brain-single.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/norris-2001-implications-bulk-motion-diffusionweighted-imaging-experim.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/atkinson-2000-sampling-reconstruction-effects-due-motion-diffusionweig.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/trouard-1996-analysis-comparison-motioncorrection-techniques-diffusion.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/anderson-1994-analysis-correction-motion-artifacts-diffusion-weighted.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/wedeen-1994-mri-signal-void-due-inplane-motion.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/johansenberg-2014-diffusion-mri-quantitative-measurement-vivo-neuroana.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/jones-2011-diffusion-mri-theory-methods-applications.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/anon-2010-tortoise-integrated-software-package-processing-diffusion.md |  |
| refs/andersson-2016-outlier-replacement.md | cites | refs/cook-2006-camino-open-source-diffusion-mri-reconstruction-processing.md |  |
| refs/yeh-2010-generalized-q-sampling-imaging.md | informs | pages/entity/concept-generalized-q-sampling-imaging.md | theory grounding |
| refs/yeh-2010-generalized-q-sampling-imaging.md | extends | refs/tuch-2004-q-ball-imaging.md | QBI is the infinite-L limit of GQI |
| pages/entity/concept-generalized-q-sampling-imaging.md | depends_on | pages/entity/concept-diffusion-mri-signal.md | inherits q-space / signal symbols |
| pages/synthesis/q-space-reconstruction-landscape.md | depends_on | refs/yeh-2010-generalized-q-sampling-imaging.md | GQI unifies the landscape |
| pages/synthesis/q-space-reconstruction-landscape.md | cites | refs/tuch-2004-q-ball-imaging.md | QBI shell method |
| pages/synthesis/q-space-reconstruction-landscape.md | informs | pages/entity/concept-generalized-q-sampling-imaging.md | positions GQI vs QBI/DSI |
