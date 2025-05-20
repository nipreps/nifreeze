*NiFreeze*
==========

**Model-based estimation and correction of head motion and eddy current distortions in 4D neuroimaging**.

*NiFreeze* is a flexible framework for volume-to-volume motion estimation and correction in d/fMRI and PET,
and eddy-current-derived distortion estimation in dMRI.

.. important:: *NiFreeze* is a fork of *eddymotion*

   In November 2024, the *NiPreps Steering Committee* brought to the *Bi-monthly Roundup*
   the discussion about re-branding *eddymotion* to better reflect its aspirations to
   perform on diverse modalities.

   The `repository of the project <https://github.com/nipreps/eddymotion>`__
   has been archived, and development will continue under the *NiFreeze* project.

   The contributor list of *eddymotion* is found under the credit file
   ``.maint/EDDYMOTION.md`` in this repository.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4680599.svg
   :target: https://doi.org/10.5281/zenodo.4680599
   :alt: DOI

.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
   :target: https://github.com/nipreps/nifreeze/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/pypi/v/nifreeze.svg
   :target: https://pypi.python.org/pypi/nifreeze/
   :alt: Latest Version

.. image:: https://github.com/nipreps/nifreeze/actions/workflows/test.yml/badge.svg
   :target: https://github.com/nipreps/nifreeze/actions/workflows/test.yml
   :alt: Testing

.. image:: https://github.com/nipreps/nifreeze/actions/workflows/notebooks.yml/badge.svg
   :target: https://github.com/nipreps/nifreeze/actions/workflows/notebooks.yml
   :alt: Examples

.. image:: https://github.com/nipreps/nifreeze/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://www.nipreps.org/nifreeze/main/index.html
   :alt: Documentation

.. image:: https://github.com/nipreps/nifreeze/actions/workflows/pythonpackage.yml/badge.svg
   :target: https://github.com/nipreps/nifreeze/actions/workflows/pythonpackage.yml
   :alt: Python package

.. image:: https://github.com/nipreps/nifreeze/actions/workflows/contrib.yml/badge.svg
   :target: https://github.com/nipreps/nifreeze/actions/workflows/contrib.yml
   :alt: Contribution checks

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
   :alt: Code format

Diffusion and functional MRI (d/fMRI) generally employ echo-planar imaging (EPI) for fast
whole-brain acquisition.
Despite the rapid collection of volumes, typical repetition times are long enough for head motion
to occur, which has been proven detrimental to both diffusion [1]_ and functional [2]_ MRI.
In the case of dMRI, additional volume-wise, low-order spatial distortions are caused by
eddy currents (EC), which appear as a result of quickly switching diffusion gradients.
Unaccounted for EC distortion can result in incorrect local model fitting and poor downstream
tractography results [3]_, [4]_.
*FSL*'s ``eddy`` [5]_ is the most popular tool for EC distortion correction, and
implements a leave-one-volume-out approach to estimate EC distortions.
However, *FSL* has commercial restrictions that hinder application within open-source initiatives
such as *NiPreps* [6]_.
In addition, *FSL*'s development model discourages the implementation of alternative data-modeling
approaches to broaden the scope of application (e.g., modalities beyond dMRI).
*NiFreeze* is an open-source implementation of ``eddy``'s approach to estimate artifacts
that permits alternative models that apply to, for instance, head motion estimation in fMRI 
and positron-emission tomography (PET) data.

.. BEGIN FLOWCHART

.. image:: https://raw.githubusercontent.com/nipreps/nifreeze/main/docs/_static/nifreeze-flowchart.png
   :alt: The nifreeze flowchart

.. END FLOWCHART

.. [1] Yendiki et al. (2014) *Spurious group differences due to head motion in a diffusion MRI study*.
    NeuroImage **88**:79-90.

.. [2] Power et al. (2012) *Spurious but systematic correlations in functional connectivity MRI
    networks arise from subject motion*. NeuroImage **59**:2142-2154.

.. [3] Zhuang et al. (2006) *Correction of eddy-current distortions in diffusion tensor images using
    the known directions and strengths of diffusion gradients*. J Magn Reson Imaging **24**:1188-1193.

.. [4] Andersson et al. (2012) *A comprehensive Gaussian Process framework for correcting distortions
    and movements in diffusion images*. In: 20th SMRT & 21st ISMRM, Melbourne, Australia.

.. [5] Andersson & Sotiropoulos (2015) *Non-parametric representation and prediction of single- and
    multi-shell diffusion-weighted MRI data using Gaussian processes*. NeuroImage **122**:166-176.

.. [6] Esteban (2025) *Standardized preprocessing in neuroimaging: enhancing reliability and reproducibility*.
    In: Whelan, R., & Lemaître, H. (eds.) *Methods for Analyzing Large Neuroimaging Datasets. Neuromethods*,
    vol. **218**, pp. 153-179. Humana, New York, NY.
    doi:`10.1007/978-1-0716-4260-3_8 <https://doi.org/10.1007/978-1-0716-4260-3_8>`__.
