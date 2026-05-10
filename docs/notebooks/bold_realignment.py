#!/usr/bin/env python
# coding: utf-8

# In[16]:


import asyncio
from os import getenv
from pathlib import Path
from shutil import copy, move

import nest_asyncio
import nibabel as nb
import nitransforms as nt
import numpy as np
from nipreps.synthstrip.wrappers.nipype import SynthStrip
from nipype.interfaces.afni import Volreg
from scipy.ndimage import binary_dilation
from skimage.morphology import ball

from nifreeze.registration import ants as erants

nest_asyncio.apply()


# In[17]:


# Install test data from gin.g-node.org:
#   $ datalad install -g https://gin.g-node.org/nipreps-data/tests-nifreeze.git
# and point the environment variable TEST_DATA_HOME to the corresponding folder
DATA_PATH = Path(getenv("TEST_DATA_HOME", str(Path.home() / "nifreeze-tests")))
WORKDIR = Path.home() / "tmp" / "nifreezedev" / "ismrm25"
WORKDIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path.home() / "tmp" / "nifreezedev" / "ismrm25" / "nifreeze-ismrm25-exp2"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


async def main():

    # In[18]:


    bold_runs = [
        Path(line)
        for line in (DATA_PATH / "ismrm_sample.txt").read_text().splitlines()
        if line.strip()
    ]


    # In[20]:


    results = []
    for bold_run in bold_runs:
        avg_path = OUTPUT_DIR / bold_run.parent / f"{bold_run.name.rsplit('_', 1)[0]}_boldref.nii.gz"

        if not avg_path.exists():
            nii = nb.load(DATA_PATH / bold_run)
            average = nii.get_fdata().mean(-1)
            avg_path.parent.mkdir(exist_ok=True, parents=True)
            nii.__class__(average, nii.affine, nii.header).to_filename(avg_path)

        bmask_path = (
            OUTPUT_DIR / bold_run.parent / f"{bold_run.name.rsplit('_', 1)[0]}_label-brain_mask.nii.gz"
        )
        print(f"Data path: {DATA_PATH}")
        for item in DATA_PATH.iterdir():
            print(f"Item: {item}")

        if not DATA_PATH.exists() or not DATA_PATH.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {DATA_PATH}")

        import hashlib
        model_path = DATA_PATH / "synthstrip.1.pt"
        md5_hash = hashlib.md5()
        try:
            with open(model_path, "rb") as file:
                # Read the file in chunks to handle large files efficiently
                for chunk in iter(lambda: file.read(4096), b""):
                    md5_hash.update(chunk)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found.")

        md5_value = md5_hash.hexdigest()
        print(f"The MD5 hash of {model_path} is: {md5_value}")
        import os
        use_gpu = os.getenv("HAS_CUDA", "false").lower() == "true"
        print(f"Use GPU: {use_gpu}")
        if not bmask_path.exists():
            bmsk_results = SynthStrip(
                in_file=str(avg_path),
                use_gpu=use_gpu,
                model=model_path,
            ).run(cwd=str(WORKDIR))

            ssiface = SynthStrip(
                in_file=str(avg_path),
                use_gpu=use_gpu,
            )
            print(ssiface.cmdline)
            bmsk_results = ssiface.run(cwd=str(WORKDIR))
            print(f"Outputs: {bmsk_results.outputs}")
            print(f"Out_mask: {bmsk_results.outputs.out_mask}")
            copy(bmsk_results.outputs.out_mask, bmask_path)

        dilmask_path = (
            avg_path.parent / f"{avg_path.name.rsplit('_', 1)[0]}_label-braindilated_mask.nii.gz"
        )

        if not dilmask_path.exists():
            niimsk = nb.load(bmask_path)
            niimsk.__class__(
                binary_dilation(niimsk.get_fdata() > 0.0, ball(4)).astype("uint8"),
                niimsk.affine,
                niimsk.header,
            ).to_filename(dilmask_path)

        oned_matrix_path = avg_path.parent / f"{avg_path.name.rsplit('_', 1)[0]}_desc-hmc_xfm.txt"
        realign_output = (
            avg_path.parent / f"{avg_path.name.rsplit('_', 1)[0]}_desc-realigned_bold.nii.gz"
        )

        if not realign_output.exists():
            volreg_results = Volreg(
                in_file=str(DATA_PATH / bold_run),
                in_weight_volume=str(dilmask_path),
                args="-Fourier -twopass",
                zpad=4,
                outputtype="NIFTI_GZ",
                oned_matrix_save=f"{oned_matrix_path}.aff12.1D",
                out_file=str(realign_output),
                num_threads=12,
            ).run(cwd=str(WORKDIR))

            move(volreg_results.outputs.oned_matrix_save, oned_matrix_path)


    # In[ ]:


    afni_realigned = [
        OUTPUT_DIR / bold_run.parent / f"{bold_run.name.rsplit('_', 1)[0]}_desc-realigned_bold.nii.gz"
        for bold_run in bold_runs
    ]


    # In[ ]:


    afni_realigned


    # In[ ]:


    from matplotlib import pyplot as plt


    # In[ ]:


    def plot_profile(image_path, axis=None, indexing=None, cmap="gray", label=None, figsize=(15, 1.7)):
        """Plots a single image slice on a given axis or a new figure if axis is None."""
        # Load the image
        image_data = nb.load(image_path).get_fdata()

        # Define default indexing if not provided
        if indexing is None:
            indexing = (
                image_data.shape[0] // 2,
                3 * image_data.shape[1] // 4,
                slice(None),
                slice(None),
            )

        # If no axis is provided, create a new figure and axis
        if axis is None:
            fig, axis = plt.subplots(figsize=figsize)
        else:
            fig = None  # If axis is provided, we won't manage the figure

        # Display the image on the specified axis with aspect='auto' and the colormap
        axis.imshow(image_data[indexing], aspect="auto", cmap=cmap)

        # Turn off the axis for a cleaner look
        axis.axis("off")

        if label:
            # Annotate the plot with the provided label
            axis.text(
                0.02,
                0.95,
                label,
                color="white",
                fontsize=12,
                ha="left",
                va="top",
                transform=axis.transAxes,
            )

        # If we created the figure, show it
        if fig is not None:
            plt.show()

        return fig


    # def plot_combined_profile(images, indexing=None, figsize=(15, 1.7), cmap='gray', labels=None):
    #     # Create a figure with three subplots in a vertical layout and specified figure size
    #     n_images = len(images)

    #     nplots = n_images * len(indexing or [True])
    #     figsize = (figsize[0], figsize[1] * nplots)
    #     fig, axes = plt.subplots(nplots, 1, figsize=figsize, constrained_layout=True)

    #     if labels is None or isinstance(labels, str):
    #         labels = (labels, ) * nplots

    #     if indexing is None or len(indexing) == 0:
    #         indexing = [None]

    #     for i, idx in enumerate(indexing):
    #         for j in range(len(images)):
    #             ax = axes[i * n_images + j]
    #             plot_profile(images[j], axis=ax, indexing=idx, cmap=cmap, label=labels[j])

    #     return fig


    def plot_combined_profile(
        images, _afni_fd, _nifreeze_fd, indexing=None, figsize=(15, 1.7), cmap="gray", labels=None
    ):
        # Calculate the number of profile plots
        n_images = len(images)
        nplots = n_images * len(indexing or [True])
        total_height = figsize[1] * nplots + 2  # Adjust figure height for FD plot

        # Create a figure with one extra row for the FD plot, setting `sharex=True` for shared x-axis
        fig, axes = plt.subplots(
            nplots + 1, 1, figsize=(figsize[0], total_height), constrained_layout=True, sharex=True
        )

        # Plot the framewise displacement on the first axis
        fd_axis = axes[0]
        timepoints = np.arange(len(_afni_fd))  # Assuming afni_fd and nifreeze_fd have the same length
        fd_axis.plot(timepoints, _afni_fd, label="AFNI 3dVolreg FD", color="blue")
        fd_axis.plot(timepoints, _nifreeze_fd, label="nifreeze FD", color="orange")
        fd_axis.set_ylabel("FD (mm)")
        fd_axis.legend(loc="upper right")
        fd_axis.set_xticks([])  # Hide x-ticks to keep x-axis clean

        # Set labels for profile plots if not provided
        if labels is None or isinstance(labels, str):
            labels = (labels,) * nplots

        # Set indexing if not provided
        if indexing is None or len(indexing) == 0:
            indexing = [None]

        # Plot each profile slice below the FD plot
        for i, idx in enumerate(indexing):
            for j in range(len(images)):
                ax = axes[i * n_images + j + 1]  # Shift index by 1 to account for FD plot
                plot_profile(images[j], axis=ax, indexing=idx, cmap=cmap, label=labels[j])

        return fig


    # In[ ]:


    # plot_combined_profile(
    #     (DATA_PATH / bold_runs[15], afni_realigned[15], afni_realigned[15]),
    #     afni_fd, nifreeze_fd,
    #     labels=("hmc1", "original", "hmc2"),
    # )


    # In[ ]:


    datashape = nb.load(DATA_PATH / bold_runs[15]).shape
    #plot_profile(
    #    DATA_PATH / bold_runs[15],
    #    afni_realigned[15],
    #    afni_realigned[15],
    #    indexing=(slice(None), 3 * datashape[1] // 4, datashape[2] // 2, slice(None)),
    #);


    # In[ ]:


    from nifreeze.model.base import ExpectationModel
    from nifreeze.utils.iterators import random_iterator


    # In[ ]:


    async def ants(t, data, hdr, nii, brainmask_path, semaphore, workdir):
        async with semaphore:
            # Set up paths
            fixed_path = workdir / f"fixedimage_{t:04d}.nii.gz"
            moving_path = workdir / f"movingimage_{t:04d}.nii.gz"

            # Create a mask for the specific timepoint
            t_mask = np.zeros(data.shape[-1], dtype=bool)
            t_mask[t] = True

            # Fit and predict using the model
            model = ExpectationModel()
            model.fit(
                data[..., ~t_mask],
                stat="median",
            )
            fixed_data = model.predict()

            # Save fixed and moving images
            nii.__class__(fixed_data, nii.affine, hdr).to_filename(fixed_path)
            nii.__class__(data[..., t_mask], nii.affine, hdr).to_filename(moving_path)

            # Generate the command
            cmdline = erants.generate_command(
                fixed_path,
                moving_path,
                fixedmask_path=brainmask_path,
                output_transform_prefix=f"conversion-{t:02d}",
                num_threads=8,
            ).cmdline

            # Run the command
            proc = await asyncio.create_subprocess_shell(
                cmdline,
                cwd=str(workdir),
                stdout=(workdir / f"ants-{t:04d}.out").open("w+"),
                stderr=(workdir / f"ants-{t:04d}.err").open("w+"),
            )
            returncode = await proc.wait()
            return returncode


    # In[ ]:


    # Set up concurrency limit and tasks
    semaphore = asyncio.Semaphore(12)
    tasks = []

    # Load and preprocess data
    for bold_run in bold_runs:
        print(bold_run.parent)
        workdir = WORKDIR / bold_run.parent
        workdir.mkdir(parents=True, exist_ok=True)
        data_path = DATA_PATH / bold_run
        brainmask_path = (
            OUTPUT_DIR / bold_run.parent / f"{bold_run.name.rsplit('_', 1)[0]}_label-brain_mask.nii.gz"
        )

        nii = nb.load(data_path)
        hdr = nii.header.copy()
        hdr.set_sform(nii.affine, code=1)
        hdr.set_qform(nii.affine, code=1)
        data = nii.get_fdata(dtype="float32")
        n_timepoints = data.shape[-1]

        # Start tasks immediately upon creation
        for t in random_iterator(n_timepoints):  # Random iterator
            task = asyncio.create_task(ants(t, data, hdr, nii, brainmask_path, semaphore, workdir))
            tasks.append(task)

    # Await all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)


    # In[ ]:


    from nitransforms.resampling import apply

    from nifreeze.registration.utils import displacement_framewise

    afni_fd = {}
    nitransforms_fd = {}
    # Load and preprocess data
    for bold_run in bold_runs:
        work_path = WORKDIR / bold_run.parent

        n_timepoints = len(list(work_path.glob("*.mat")))

        xfms = [
            nt.linear.Affine(
                nt.io.itk.ITKLinearTransform.from_filename(
                    work_path / f"conversion-{t:02d}0GenericAffine.mat"
                ).to_ras(
                    reference=work_path / f"fixedimage_{t:04d}.nii.gz",
                    moving=work_path / f"movingimage_{t:04d}.nii.gz",
                )
            )
            for t in range(n_timepoints)
        ]

        nii = nb.load(DATA_PATH / bold_run)
        nitransforms_fd[str(bold_run)] = np.array([displacement_framewise(nii, xfm) for xfm in xfms])

        hmc_xfm = nt.linear.LinearTransformsMapping(xfms)
        out_nitransforms = (
            OUTPUT_DIR
            / bold_run.parent
            / f"{bold_run.name.rsplit('_', 1)[0]}_desc-nitransforms_bold.nii.gz"
        )
        if not out_nitransforms.exists():
            apply(
                hmc_xfm,
                spatialimage=nii,
                reference=nii,
            ).to_filename(out_nitransforms)

        afni_xfms = nt.linear.load(
            OUTPUT_DIR / bold_run.parent / f"{bold_run.name.rsplit('_', 1)[0]}_desc-hmc_xfm.txt"
        )
        afni_fd[str(bold_run)] = np.array(
            [displacement_framewise(nii, afni_xfms[i]) for i in range(len(afni_xfms))]
        )

        out_afni = (
            OUTPUT_DIR / bold_run.parent / f"{bold_run.name.rsplit('_', 1)[0]}_desc-afni_bold.nii.gz"
        )
        if not out_afni.exists():
            apply(
                afni_xfms,
                spatialimage=nii,
                reference=nii,
            ).to_filename(out_afni)


    # In[ ]:


    afni_fd


    # In[ ]:


    # Generate an index.html with links to each SVG file
    index_path = OUTPUT_DIR / "index.html"
    with open(index_path, "w") as index_file:
        index_file.write("<html><body>\n")
        index_file.write("<h1>Profile Plot Index</h1>\n<ul>\n")

        for bold_run in bold_runs:
            original = DATA_PATH / bold_run
            nitransforms = (
                OUTPUT_DIR
                / bold_run.parent
                / f"{bold_run.name.rsplit('_', 1)[0]}_desc-nitransforms_bold.nii.gz"
            )
            afni = (
                OUTPUT_DIR
                / bold_run.parent
                / f"{bold_run.name.rsplit('_', 1)[0]}_desc-realigned_bold.nii.gz"
            )

            datashape = nb.load(original).shape

            fig = plot_combined_profile(
                (afni, original, nitransforms),
                afni_fd[str(bold_run)],
                nitransforms_fd[str(bold_run)],
                labels=("3dVolreg", str(bold_run), "nifreeze"),
                indexing=(None, (slice(None), 3 * datashape[1] // 4, datashape[2] // 2, slice(None))),
            )

            # Save the figure
            out_svg = OUTPUT_DIR / bold_run.parent / bold_run.name.replace(".nii.gz", ".svg")
            fig.savefig(out_svg, format="svg")
            fig.savefig(out_svg.with_suffix(".png"), format="png", dpi=320)
            plt.close(fig)

            index_file.write(f"<li><a href={out_svg.relative_to(OUTPUT_DIR)}>{bold_run}</a></li>\n")

        index_file.write("</ul>\n</body></html>")


    # In[ ]:


if __name__ == "__main__":
    asyncio.run(main())
