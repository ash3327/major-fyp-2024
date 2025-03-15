# KTL2401 General Hand Gesture Recognition

## Quick Links
* **Resources:** [Datasets](/docs/README-datasets.md) | [Dataset Download Instructions](/docs/README-folders-datasets.md) | [Papers](docs/README-papers.md) 
* **Drafts:** [Planning Document](https://1drv.ms/w/s!Ago9nLnz9h82gosJqjlnjoOBK9SS-Q?e=YuU6Wy) | [LaTeX Drafts](https://www.overleaf.com/6774289999vrsjksvmvfch#f96db3)
* **Reports:** [Planning Report](/docs/KTL2401_1155175983_1155174636_planning_report.pdf) | 
[First Term Report](/docs/KTL2401_1155175983_1155174636_final_report.pdf)
* **Progress:** [Progress Document](/docs/README-progress.md)

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
  * 

### Data Extraction

* **Code**:  `python prepare_dataset.py <dataset_name>`.
* **Steps**:
    1. 


