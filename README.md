# KTL2401 General Hand Gesture Recognition

## Quick Links
* **Resources:** [Datasets](/docs/README-datasets.md) | [Dataset Download Instructions](/docs/README-folders-datasets.md) | [Papers](docs/README-papers.md) 
* **Drafts:** [Planning Document](https://1drv.ms/w/s!Ago9nLnz9h82gosJqjlnjoOBK9SS-Q?e=YuU6Wy) | [LaTeX Drafts](https://www.overleaf.com/6774289999vrsjksvmvfch#f96db3)
* **Reports:** [Planning Report](/docs/KTL2401_1155175983_1155174636_planning_report.pdf) | 
[First Term Report](/docs/KTL2401_1155175983_1155174636_final_report.pdf)
* **Progress:** [Progress Document](/docs/README-progress.md)

> [!NOTE]
> Please read the [Progress Document](/docs/README-progress.md) to see the current progress.

## Goals

* **Adaptation.** Allowing for quick adaptation to new set of data without full-scale re-training.

## Approach

* **Contrastive Pre-training.** Utilizes contrastive learning to 

## Setting Up

### Environment

* **System Requirement:**
    * Requires: Python 3.10, CUDA 11.8, CUDNN 8
* **Virtual Environment:** 
    1. Create and enter virtual environment (expect this will require around 15 minutes).
        ```bash
        # on windows cmd
        virtualenv -p python3.10 venv # create venv
        .\venv\Scripts\activate # enter venv

        # on wsl/linux
        python3.10 -m venv env # create env
        source env/bin/activate # enter env

        # please remember to set your default interpreter to venv or env.
        ```
    2. `pip install -r requirements.txt` 
* **Data Architecture:**
  * All data are stored within the directory `data/raw` and the processed data are stored under `data/kpts`.
  * Download the datasets according to [this document](/docs/README-folders-datasets.md).

### Data Extraction

* For the following datasets, perform `python scripts/datasets/prepare_dataset.py <dataset_name>`, where `<dataset_name>` is one of the followings:
  * Static: `lexset` (`synthetic-asl-alphabet`), `senz3d` (`senz3d_dataset`), `handshape` (`ph2014-handshape`).
* Fixing: `python scripts/datasets/_fix_holistic.py -d <dataset_name> -s <split_name>` if there are two hands within the video guaranteeed.
* Fixing: `python scripts/datasets/_fix_left_right_hands.py -d <dataset_name> -s <split_name>`
* Verification: `python scripts/datasets/dataset_loader.py -d <dataset_name> -s <split_name>`

## Training

### Folder Architecture

- Dataset preparation: under `scripts/datasets`.
- Dataset for hand-only contrastive training: under `scripts/fake_data` and `scripts/hand_only_supervised`.
- Unsupervised contrastive training (InfoNCE loss) (v3): `python training/contrastive/train_contrastive_fake_data.py`.
