# Test
mkdir -p "data/raw/ph2014-handshape/test"

# Fetch the file
wget https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/ph2014-dev-set-handshape-annotations.tar.gz -O data/raw/ph2014-dev-set-handshape-annotations.tar.gz

# No progress bar
# tar -xvf data/raw/ph2014-dev-set-handshape-annotations.tar.gz -C data/raw/ph2014-handshape/test --strip-components=1

# or WSL
pv data/raw/ph2014-dev-set-handshape-annotations.tar.gz | tar -xzf - -C data/raw/ph2014-handshape/test --strip-components=1
# 
# Replace `test` with `train` and the tar path with the train dataset path to unpack the train dataset.