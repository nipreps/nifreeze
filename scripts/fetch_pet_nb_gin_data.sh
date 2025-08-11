#!/bin/bash

DEST_DIR=$1

# Define variables
LIST_URL="gin.g-node.org/nipreps-data/tests-nifreeze/raw/master"

FOLDER="pet_data"
SRC_SUB_LABELS=("sub-01")
MTN_SUB_LABELS=("sub-02")
SUB_LABELS=("${SRC_SUB_LABELS[@]}" "${MTN_SUB_LABELS[@]}")
SESSION_LABEL="ses-baseline"
ANAT_LABEL="anat"
PET_LABEL="pet"
ANAT_FNAMES_CMN_SSTR=("ses-baseline_T1w.nii")
PET_FNAMES_CMN_SSTR=("ses-baseline_pet.json" "ses-baseline_pet.nii.gz" "ses-baseline_recording-manual_blood.json" "ses-baseline_recording-manual_blood.tsv")
MOTION_FNAMES_CMN_SSTR=("ses-baseline_ground_truth_motion.csv")
UNDERSCORE="_"

# Create target directory structure
for sub_id in "${SUB_LABELS[@]}"; do
    mkdir -p "${DEST_DIR}/${FOLDER}/${sub_id}/${SESSION_LABEL}/${ANAT_LABEL}"
    mkdir -p "${DEST_DIR}/${FOLDER}/${sub_id}/${SESSION_LABEL}/${PET_LABEL}"
done

# Download anatomical files
for sub_id in "${SUB_LABELS[@]}"; do
    fname="${ANAT_FNAMES_CMN_SSTR[@]/#/${sub_id}${UNDERSCORE}}"
    url="${LIST_URL}/${FOLDER}/${sub_id}/${SESSION_LABEL}/${ANAT_LABEL}/${fname}"
    wget -nv -O "${DEST_DIR}/${FOLDER}/${sub_id}/${SESSION_LABEL}/${ANAT_LABEL}/${fname}" "${url}"
done

# Download PET files
for sub_id in "${SUB_LABELS[@]}"; do
    fnames=("${PET_FNAMES_CMN_SSTR[@]/#/${sub_id}${UNDERSCORE}}")
    for fname in "${fnames[@]}"; do
      url="${LIST_URL}/${FOLDER}/${sub_id}/${SESSION_LABEL}/${PET_LABEL}/${fname}"
      wget -nv -O "${DEST_DIR}/${FOLDER}/${sub_id}/${SESSION_LABEL}/${PET_LABEL}/${fname}" "${url}"
    done
done

# Download motion files
for sub_id in "${MTN_SUB_LABELS[@]}"; do
    fnames=("${MOTION_FNAMES_CMN_SSTR[@]/#/${sub_id}${UNDERSCORE}}")
    for fname in "${fnames[@]}"; do
      url="${LIST_URL}/${FOLDER}/${SUB_LABEL}/${SESSION_LABEL}/${PET_LABEL}/${fname}"
      wget -nv -O "${DEST_DIR}/${FOLDER}/${SUB_LABEL}/${SESSION_LABEL}/${PET_LABEL}/${fname}" "${url}"
    done
done

echo "PET data successfully downloaded to the '${DEST_DIR}/${FOLDER}' directory."
