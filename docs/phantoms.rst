.. include:: links.rst

Synthetic reference data
========================
The sensitivity benchmarks compare model predictions on an aligned dataset with
predictions on copies of it whose volumes moved or were distorted by known
amounts.
Real data cannot serve as the aligned reference: they carry motion and
eddy-current distortions of their own, as large as the smallest perturbations
the benchmarks add.
*NiFreeze* therefore simulates its reference data with
`TRXScan <https://github.com/PennLINC/TRXScan>`__ (a port of MITK Fiberfox) from
one phantom anatomy, under the acquisition protocols of real studies, and
simulates every perturbation with its ground truth
(`#548 <https://github.com/nipreps/nifreeze/issues/548>`__).

Acquisition protocols
---------------------
The protocols are chosen to span the axes along which models differ most, while
standing for data that *NiFreeze* users actually process:

- the number of shells and the largest b-value (one to four shells, b=1000 to 3500);
- the number of directions per shell and the coverage of the sphere (32 to 150
  directions; half or whole sphere, which decides whether eddy-current
  distortions cancel over the directions);
- the voxel size and the signal-to-noise ratio it implies (1.25 to 2 mm);
- prevalence: the protocols behind large public datasets.

.. list-table::
   :header-rows: 1

   * - Protocol
     - Shells (directions)
     - b=0
     - Voxel (mm)
     - Scanner
     - Stands for
   * - HCP-YA
     - 1000, 2000, 3000 (90 each)
     - 18
     - 1.25
     - Siemens Skyra Connectom, 3 T
     - Dense, balanced multi-shell research data
   * - Healthy Brain Network
     - 1000, 2000 (64 each)
     - 1
     - 1.8
     - Siemens Prisma, 3 T
     - Large pediatric two-shell cohorts
   * - ABCD
     - 500, 1000, 2000, 3000 (6, 15, 15, 60)
     - 7
     - 1.7
     - Siemens Prisma, 3 T
     - Unbalanced shells with sparse inner shells
   * - Sherbrooke
     - 1000, 2000, 3500 (64 each)
     - 1
     - 2
     - Siemens, 1.5 T
     - The highest b-value; DIPY's multi-shell example
   * - Stanford HARDI
     - 2000 (150)
     - 10
     - 2
     - GE MR750, 3 T
     - High angular resolution on one shell
   * - AOMIC-PIOP2
     - 1000 (32, half sphere)
     - 1
     - 2
     - Philips, 3 T
     - The minimal clinical acquisition

``tools/phantoms/protocols.toml`` records, for each protocol, its gradient table
(``tools/phantoms/protocols/<name>.b``: one ``x y z b`` row per volume, unit
vectors in world RAS coordinates as prescribed), the source of the table with its
checksum, license and citation, the acquisition parameters the simulation needs,
and the signal-to-noise ratio of the b=0 images.
``tools/phantoms/fetch_protocols.py`` rebuilds the tables from their sources,
and its ``--check`` option compares them with the stored ones.
``tools/phantoms/measure_snr.py`` measures the signal-to-noise ratio on the
real data of each protocol, where they are openly available: from 8.9
(Sherbrooke, at 1.5 T) to 19.4 (AOMIC-PIOP2).
The values of HCP-YA (9.0) and ABCD (13.4), whose data are not openly
available, are extrapolated from published or measured ones, as the catalog
records.

Sources come in different frames, all converted to world RAS coordinates.
The b-vectors of HBN and AOMIC are converted with the header of the image they
came with; those of ABCD, Sherbrooke and Stanford are taken in the voxel frame of
an image aligned with the world axes (LAS for ABCD, as ``dcm2niix`` writes the
axial images of the same Siemens sites).
Diffusion contrast does not depend on the sign of a b-vector, so these
assumptions only affect the sign of simulated eddy-current distortions.
Three tables share one set of 64 directions, up to axis flips: those of HBN,
Sherbrooke and the single-shell dataset ``ds000114``, which is why the clinical
protocol is AOMIC-PIOP2 instead of the latter.
