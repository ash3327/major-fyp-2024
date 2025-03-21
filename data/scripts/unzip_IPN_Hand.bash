# Define the input directory containing the zip files
INPUT_DIR="data/raw"
# Define the output directory where the extracted files will be placed
OUTPUT_DIR="data/raw"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all zip files in the input directory
for zip_file in "$INPUT_DIR"/IPN_Hand-20250320T111646Z-*.zip; do
    # Extract the base name of the zip file (e.g., "IPN_Hand-20250320T111646Z-001")
    base_name=$(basename "$zip_file" .zip)
    
    # Extract the zip file into the corresponding folder within the output directory
    7z x "$zip_file" -o"$OUTPUT_DIR" -aos

    # Check if the unzip was successful
    if [ $? -eq 0 ]; then
        echo "Unzip of $zip_file completed successfully."
    else
        echo "Unzip of $zip_file failed."
        exit 1
    fi
done

#!/bin/bash

# Define the base directory containing the .tgz files
BASE_DIR="data/raw/IPN_Hand"

# Function to extract .tgz files in a given directory
extract_tgz_files() {
    local dir=$1
    for tgz_file in "$dir"/*.tgz; do
        if [ -f "$tgz_file" ]; then
            echo "Extracting $tgz_file..."
            # Use pv for a progress bar and suppress intermediate file names
            pv "$tgz_file" | tar -xzf - -C "$BASE_DIR"
            # Check if the extraction was successful
            if [ $? -eq 0 ]; then
                echo "Extraction of $tgz_file completed successfully."
            else
                echo "Extraction of $tgz_file failed."
                exit 1
            fi
        fi
    done
}

# Extract .tgz files in the frames directory
FRAMES_DIR="$BASE_DIR/frames"
extract_tgz_files "$FRAMES_DIR"

# Extract .tgz files in the videos directory
VIDEOS_DIR="$BASE_DIR/videos"
extract_tgz_files "$VIDEOS_DIR"

echo "All .tgz files have been extracted successfully."

