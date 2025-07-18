#!/bin/bash

DEST_DIR=$1

# Define variables
LIST_URL="gin.g-node.org/jhlegarreta/tests-nifreeze/raw/AddPETData"

FOLDER="pet_data"
SUB_LABEL="sub-01"
SESSION_LABEL="ses-baseline"
ANAT_LABEL="anat"
PET_LABEL="pet"
ANAT_FNAMES=("sub-01_ses-baseline_T1w.nii")
PET_FNAMES=("sub-01_ses-baseline_pet.json" "sub-01_ses-baseline_pet.nii.gz" "sub-01_ses-baseline_recording-manual_blood.json" "sub-01_ses-baseline_recording-manual_blood.tsv")

# Create target directory structure
mkdir -p "${DEST_DIR}/${FOLDER}/${SUB_LABEL}/${SESSION_LABEL}/${ANAT_LABEL}"
mkdir -p "${DEST_DIR}/${FOLDER}/${SUB_LABEL}/${SESSION_LABEL}/${PET_LABEL}"

# Download anatomical files
for fname in "${ANAT_FNAMES[@]}"; do
    url="${LIST_URL}/${FOLDER}/${SUB_LABEL}/${SESSION_LABEL}/${ANAT_LABEL}/${fname}"
    wget -nv -O "${DEST_DIR}/${FOLDER}/${SUB_LABEL}/${SESSION_LABEL}/${ANAT_LABEL}/${fname}" "${url}"
done

# Download PET files
for fname in "${PET_FNAMES[@]}"; do
    url="${LIST_URL}/${FOLDER}/${SUB_LABEL}/${SESSION_LABEL}/${PET_LABEL}/${fname}"
    wget -nv -O "${DEST_DIR}/${FOLDER}/${SUB_LABEL}/${SESSION_LABEL}/${PET_LABEL}/${fname}" "${url}"
done

echo "PET data successfully downloaded to the '${DEST_DIR}/${FOLDER}' directory."
