# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY kIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Simulate motion for PET data and generate additional subject datasets in BIDS format."""

import argparse
import os
import shutil
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation as R

from nifreeze.utils.ndimage import load_api

ANAT_LABEL = "anat"
PET_LABEL = "pet"


def get_affine_matrix(translation: tuple, rotation: tuple, voxel_sizes: tuple):
    """Build an affine matrix from translation and rotation values.

    Parameters
    ----------
    translation : :obj:`tuple`
        Translation values in mm.
    rotation : :obj:`tuple`
        Rotation values in degrees.
    voxel_sizes : :obj:`tuple`
        Voxel sizes in mm.

    Returns
    -------
    affine_matrix : :obj:`numpy.ndarray`
        Affine matrix.
    """

    rot_mat = R.from_euler("xyz", rotation, degrees=True).as_matrix()
    trans_vox = np.array(translation) / voxel_sizes
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = rot_mat
    affine_matrix[:3, 3] = trans_vox
    return affine_matrix


def simulate_pet_motion(
    base_dir: Path,
    orig_sub: str,
    session: str,
    num_subjects: int = 1,
    trans_low: float = -3.5,
    trans_high: float = 3.5,
    rot_low: float = -1.0,
    rot_high: float = 1.0,
) -> None:
    """Simulate motion for PET data and generate additional subject datasets in BIDS format.

    Motion is simulated using random uniform translations and rotations along
    the 3 axes and using the provided upper and lower bounds. PET and anatomical
    (T1w) data for ``num_subjects`` new participants are created drawing
    different motion realizations from the original subject's data for each
    frame.

    Parameters
    ----------
    base_dir : :obj:`float`
        Path to the dataset root.
    orig_sub : :obj:`str`
        Original subject ID with no motion.
    session : :obj:`str`
        Session label.
    num_subjects : :obj:`int`, optional
        Number of additional simulated subjects.
    trans_low : :obj:`float`, optional
        Lower bound to simulate translation values in mm.
    trans_high : :obj:`float`, optional
        Upper bound to simulate translation values in mm.
    rot_low : :obj:`float`, optional
        Lower bound to simulate rotation values in degrees.
    rot_high : :obj:`float`, optional
        Upper bound to simulate rotation values in degrees.
    """

    orig_pet_path = base_dir / orig_sub / session / PET_LABEL / f"{orig_sub}_{session}_pet.nii.gz"
    orig_json_path = base_dir / orig_sub / session / PET_LABEL / f"{orig_sub}_{session}_pet.json"
    orig_blood_json = (
        base_dir
        / orig_sub
        / session
        / PET_LABEL
        / f"{orig_sub}_{session}_recording-manual_blood.json"
    )
    orig_blood_tsv = (
        base_dir
        / orig_sub
        / session
        / PET_LABEL
        / f"{orig_sub}_{session}_recording-manual_blood.tsv"
    )
    orig_anat_path = base_dir / orig_sub / session / ANAT_LABEL / f"{orig_sub}_{session}_T1w.nii"

    pet_img = load_api(orig_pet_path, nb.Nifti1Image)
    data = pet_img.get_fdata()
    affine = pet_img.affine
    voxel_sizes = pet_img.header.get_zooms()[:3]
    n_frames = data.shape[3]

    for subj_idx in range(num_subjects):
        new_sub = f"sub-{subj_idx + 2:02d}"
        seed = 42 + subj_idx
        rng = np.random.default_rng(seed)

        translations = [(0, 0, 0)]
        rotations = [(0, 0, 0)]
        for _ in range(1, n_frames):
            translations.append(tuple(rng.uniform(trans_low, trans_high, size=3)))
            rotations.append(tuple(rng.uniform(rot_low, rot_high, size=3)))

        motion_data = np.zeros_like(data)
        affines = []
        for frame in range(n_frames):
            aff_matrix = get_affine_matrix(translations[frame], rotations[frame], voxel_sizes)
            affines.append(aff_matrix)
            inv_aff = np.linalg.inv(aff_matrix)
            motion_data[..., frame] = affine_transform(
                data[..., frame], inv_aff[:3, :3], inv_aff[:3, 3], order=3, mode="constant", cval=0
            )

        framewise_displacement = [0] + [
            np.linalg.norm(affines[i][:3, 3] - affines[i - 1][:3, 3]) for i in range(1, n_frames)
        ]

        new_sub_pet_dir = base_dir / new_sub / session / PET_LABEL
        new_sub_anat_dir = base_dir / new_sub / session / ANAT_LABEL
        os.makedirs(new_sub_pet_dir, exist_ok=True)
        os.makedirs(new_sub_anat_dir, exist_ok=True)

        new_pet_fname = f"{new_sub}_{session}_pet.nii.gz"
        new_pet_path = new_sub_pet_dir / new_pet_fname
        nb.save(nb.Nifti1Image(motion_data, affine, pet_img.header), new_pet_path)

        shutil.copy(orig_json_path, Path(str(new_pet_path).replace(".nii.gz", ".json")))
        shutil.copy(
            orig_blood_json, new_sub_pet_dir / f"{new_sub}_{session}_recording-manual_blood.json"
        )
        shutil.copy(
            orig_blood_tsv, new_sub_pet_dir / f"{new_sub}_{session}_recording-manual_blood.tsv"
        )
        shutil.copy(orig_anat_path, new_sub_anat_dir / f"{new_sub}_{session}_T1w.nii")

        motion_df = pd.DataFrame(
            {
                "frame": np.arange(n_frames),
                "trans_x": [t[0] for t in translations],
                "trans_y": [t[1] for t in translations],
                "trans_z": [t[2] for t in translations],
                "rot_x": [r[0] for r in rotations],
                "rot_y": [r[1] for r in rotations],
                "rot_z": [r[2] for r in rotations],
                "framewise_displacement": framewise_displacement,
            }
        )
        motion_df.to_csv(
            new_sub_pet_dir / f"{new_sub}_{session}_ground_truth_motion.csv", index=False
        )

        print(f"Successfully created simulated dataset: {new_sub}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for command-line interface.

    Returns
    -------
    :obj:`~argparse.ArgumentParser`
        Argument parser for the script.

    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        help="Base directory",
    )
    parser.add_argument(
        "orig_sub",
        type=str,
        help="Original subject id",
    )
    parser.add_argument(
        "session",
        type=str,
        help="Session",
    )
    parser.add_argument(
        "--num_subjects",
        type=int,
        default=3,
        help="Number of subjects to simulate",
    )
    parser.add_argument(
        "--trans_low",
        type=float,
        default=-3.5,
        help="Lower bound to simulate translation values in mm",
    )
    parser.add_argument(
        "--trans_high",
        type=float,
        default=3.5,
        help="Upper bound to simulate translation values in mm",
    )
    parser.add_argument(
        "--rot_low",
        type=float,
        default=-1.0,
        help="Lower bound to simulate rotation values in degrees",
    )
    parser.add_argument(
        "--rot_high",
        type=float,
        default=1.0,
        help="Upper bound to simulate rotation values in degrees.",
    )
    return parser


def _parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    parser : :obj:`~argparse.ArgumentParser`
        Argument parser for the script.

    Returns
    -------
    :obj:`~argparse.Namespace`
        Parsed arguments.
    """
    return parser.parse_args()


def main() -> None:
    """Main function for running the experiment."""
    parser = _build_arg_parser()
    args = _parse_args(parser)

    simulate_pet_motion(
        args.base_dir,
        args.orig_sub,
        args.session,
        num_subjects=args.num_subjects,
        trans_low=args.trans_low,
        trans_high=args.trans_high,
        rot_low=args.rot_low,
        rot_high=args.rot_high,
    )


if __name__ == "__main__":
    main()
