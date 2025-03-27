# Test
mkdir -p "data/raw/ph2014-handshape/train"

# Fetch the file
FILE=data/raw/1miohands-v2-training.tar.gz

if [ ! -f "$FILE" ]; then
    wget https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/1miohands-v2-training.tar.gz -O "$FILE"
else
    echo "$FILE already exists. Not downloading."
fi

# No progress bar
# tar -xvf data/raw/1miohands-v2-training.tar.gz -C data/raw/ph2014-handshape/train --strip-components=1

# or WSL
pv "$FILE" | tar -xzf - -C data/raw/ph2014-handshape/train

# Replace `test` with `train` and the tar path with the train dataset path to unpack the train dataset.