# Contrastive Learning

This GitHub repository aims to act as a practice on various contrastive learning techniques.

Helpful resources: https://www.v7labs.com/blog/contrastive-learning-guide

## Setup

```bash
# Setup virtualenv
# Requires: Python 3.10, CUDA 11.8, CUDNN 8
# virtualenv -p python3.10 venv
# WINDOWS: .\venv\Scripts\activate
# LINUX/WSL: source venv/bin/activate

pip install -r requirements.txt
```

## Dataset

If you are training locally:
```bash
kaggle datasets download -d stoicstatic/face-recognition-dataset
mkdir "kaggle/input/face-recognition-dataset"
tar -xf face-recognition-dataset.zip -C kaggle/input/face-recognition-dataset
```
If you're training on Kaggle notebook, use [this](https://www.kaggle.com/datasets/stoicstatic/face-recognition-dataset/) dataset.

Please also change the value of the `IS_KAGGLE` parameter in the notebooks in order for the program to parse the paths correctly.

*Currently not able to run on local computer.

## Notebooks

| Notebooks | Description | Time per epoch |
| --- | --- | --- |
| copy.ipynb | runned on this machine (tailored for tf 2.10, to use GPU on Windows) | 07:58

*109 batches, batch size 128 (13952 samples)

Exploring output_dataset

Unique image shapes in: {(250, 250, 3), (128, 128, 3)}

Total number of images: 14311

Total number of people: 1680

## Methodology

### SimCLR



### KL Divergence



### Triplet Loss

