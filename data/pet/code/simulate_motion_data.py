import os
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation as R


def simulate_pet_motion(base_dir, orig_sub='sub-01', num_subjects=1, session='ses-baseline'):
    """
    Simulates motion for PET data and generates additional subject datasets in BIDS format.

    Parameters:
        base_dir (str): Path to the dataset root.
        orig_sub (str): Original subject ID with no motion (e.g., 'sub-01').
        num_subjects (int): Number of additional simulated subjects.
        session (str): Session label (default: 'ses-baseline').
    """
    orig_pet_path = os.path.join(base_dir, orig_sub, session, 'pet', f'{orig_sub}_{session}_pet.nii.gz')
    orig_json_path = os.path.join(base_dir, orig_sub, session, 'pet', f'{orig_sub}_{session}_pet.json')
    orig_blood_json = os.path.join(base_dir, orig_sub, session, 'pet', f'{orig_sub}_{session}_recording-manual_blood.json')
    orig_blood_tsv = os.path.join(base_dir, orig_sub, session, 'pet', f'{orig_sub}_{session}_recording-manual_blood.tsv')
    orig_anat_path = os.path.join(base_dir, orig_sub, session, 'anat', f'{orig_sub}_{session}_T1w.nii')

    pet_img = nib.load(orig_pet_path)
    data = pet_img.get_fdata()
    affine = pet_img.affine
    voxel_sizes = pet_img.header.get_zooms()[:3]
    n_frames = data.shape[3]

    for subj_idx in range(num_subjects):
        new_sub = f"sub-{subj_idx + 2:02d}"
        seed = 42 + subj_idx
        np.random.seed(seed)

        translations = [(0, 0, 0)]
        rotations = [(0, 0, 0)]
        for _ in range(1, n_frames):
            translations.append(tuple(np.random.uniform(-3.5, 3.5, size=3)))
            rotations.append(tuple(np.random.uniform(-1, 1, size=3)))

        def get_affine_matrix(translation_mm, rotation_deg, voxel_sizes):
            rot_mat = R.from_euler('xyz', rotation_deg, degrees=True).as_matrix()
            trans_vox = np.array(translation_mm) / voxel_sizes
            affine_matrix = np.eye(4)
            affine_matrix[:3, :3] = rot_mat
            affine_matrix[:3, 3] = trans_vox
            return affine_matrix

        motion_data = np.zeros_like(data)
        affines = []
        for frame in range(n_frames):
            aff_matrix = get_affine_matrix(translations[frame], rotations[frame], voxel_sizes)
            affines.append(aff_matrix)
            inv_aff = np.linalg.inv(aff_matrix)
            motion_data[..., frame] = affine_transform(
                data[..., frame],
                inv_aff[:3, :3],
                inv_aff[:3, 3],
                order=3,
                mode='constant',
                cval=0
            )

        framewise_displacement = [0] + [np.linalg.norm(affines[i][:3, 3] - affines[i - 1][:3, 3]) for i in range(1, n_frames)]

        new_sub_pet_dir = os.path.join(base_dir, new_sub, session, 'pet')
        new_sub_anat_dir = os.path.join(base_dir, new_sub, session, 'anat')
        os.makedirs(new_sub_pet_dir, exist_ok=True)
        os.makedirs(new_sub_anat_dir, exist_ok=True)

        new_pet_fname = f'{new_sub}_{session}_pet.nii.gz'
        new_pet_path = os.path.join(new_sub_pet_dir, new_pet_fname)
        nib.save(nib.Nifti1Image(motion_data, affine, pet_img.header), new_pet_path)

        shutil.copy(orig_json_path, new_pet_path.replace('.nii.gz', '.json'))
        shutil.copy(orig_blood_json, os.path.join(new_sub_pet_dir, f'{new_sub}_{session}_recording-manual_blood.json'))
        shutil.copy(orig_blood_tsv, os.path.join(new_sub_pet_dir, f'{new_sub}_{session}_recording-manual_blood.tsv'))
        shutil.copy(orig_anat_path, os.path.join(new_sub_anat_dir, f'{new_sub}_{session}_T1w.nii'))

        motion_df = pd.DataFrame({
            'frame': np.arange(n_frames),
            'trans_x': [t[0] for t in translations],
            'trans_y': [t[1] for t in translations],
            'trans_z': [t[2] for t in translations],
            'rot_x': [r[0] for r in rotations],
            'rot_y': [r[1] for r in rotations],
            'rot_z': [r[2] for r in rotations],
            'framewise_displacement': framewise_displacement
        })
        motion_df.to_csv(os.path.join(new_sub_pet_dir, f'{new_sub}_{session}_ground_truth_motion.csv'), index=False)

        print(f"Successfully created simulated dataset: {new_sub}")

if __name__ == "__main__":
    base_dir = '/Users/martinnorgaard/Downloads/eddymotion_pet_testdata/data'  # Update to your dataset root
    simulate_pet_motion(base_dir, orig_sub='sub-01', num_subjects=10)
