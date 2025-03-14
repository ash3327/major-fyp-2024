#!/bin/bash

# Define the URL and the output file name
URL="https://lttm.dei.unipd.it/downloads/gesture/senz3d/data/senz3d_dataset.zip"
OUTPUT_FILE="senz3d_dataset.zip"
OUTPUT_DIR="data/raw"

# Use wget to download the file
wget -O $OUTPUT_FILE $URL

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Download failed."
    exit 1
fi

# Unzip the downloaded file to the specified directory
unzip $OUTPUT_FILE -d $OUTPUT_DIR

# Check if the unzip was successful
if [ $? -eq 0 ]; then
    echo "Unzip completed successfully."
else
    echo "Unzip failed."
    exit 1
fi

# Clean up by removing the zip file
rm $OUTPUT_FILE

echo "Dataset is ready in $OUTPUT_DIR."