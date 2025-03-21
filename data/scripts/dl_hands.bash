#!/bin/bash

# Define the URL and the output file name
URL="https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/ndrczc35bt-2.zip"
OUTPUT_FILE="ndrczc35bt-2.zip"
OUTPUT_DIR="data/raw/hands_dataset_2"

# # Use wget to download the file
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

# Move everything under $OUTPUT_DIR/HANDS a dataset of static Hand-Gestures for Human-Robot Interaction/* to $OUTPUT_DIR/*
mv "$OUTPUT_DIR/HANDS a dataset of static Hand-Gestures for Human-Robot Interaction/"* $OUTPUT_DIR/

# Unzip the Subject{i} zip files
for i in {1..5}; do
    7z x $OUTPUT_DIR/Subject${i}/Subject${i}.zip -o$OUTPUT_DIR/Subject${i}
    
    # Check if the unzip was successful
    if [ $? -eq 0 ]; then
        echo "Unzip of Subject${i}.zip completed successfully."
    else
        echo "Unzip of Subject${i}.zip failed."
        # exit 1
    fi
done

echo "Dataset is ready in $OUTPUT_DIR."