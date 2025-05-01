#!/bin/bash
set -e

DEST_DIR=$1

LIST_URL="https://gin.g-node.org/nipreps-data/tests-nifreeze/raw/master/ismrm_sample.txt"
FNAME_LIST=$DEST_DIR/"ismrm_sample.txt"

# Argument check
if [ -z "$DEST_DIR" ]; then
  echo "Usage: $0 <destination_directory>"
  exit 1
fi

mkdir -p "$DEST_DIR"

# Download list file
echo "📥 Downloading file list from $LIST_URL"
curl -sSL "$LIST_URL" -o "$FNAME_LIST"

cd $DEST_DIR

# Process list
while IFS= read -r filepath; do
  # Skip empty or commented lines
  [[ -z "$filepath" || "$filepath" =~ ^# ]] && continue

  dataset=$(echo "$filepath" | cut -d/ -f1)
  dataset_dir="$DEST_DIR/$dataset"
  echo $dataset

  # Clone if not already cloned
  if [ ! -d "$dataset_dir" ]; then
    echo "🔄 Cloning dataset: $dataset"
    datalad clone "https://github.com/OpenNeuroDatasets/$dataset" "$dataset_dir"
  fi

  echo "📦 Fetching file: $filepath"
  datalad get -d "$dataset" "$filepath"

done < "$FNAME_LIST"

cd -

echo "✅ All requested files downloaded to $DEST_DIR"
