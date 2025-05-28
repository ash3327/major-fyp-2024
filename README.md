# KTL2401 General Hand Gesture Recognition

## Quick Links

> [!NOTE]
> Main Report: [Here](/docs/KTL2401_1155175983_1155174636_final_report_term2.pdf)
> 
> Other documents are drafts and should only treated as references only.

<!-- [Datasets](/docs/README-datasets.md) |  -->
* **Resources:** [Dataset Download Instructions](/docs/README-folders-datasets.md) | [Papers](docs/README-papers.md) 
* **Reports:** [Planning Report](/docs/KTL2401_1155175983_1155174636_planning_report.pdf) | 
[First Term Report](/docs/KTL2401_1155175983_1155174636_final_report_term1.pdf) | [Second Term Report (Individual)](/docs/KTL2401_1155175983_final_report_term2.pdf) | [Second Term Report (Final)](/docs/KTL2401_1155175983_1155174636_final_report_term2.pdf)
<!-- * **Progress:** [Progress Document](/docs/README-progress.md) -->

<!-- > [!NOTE]
> Please read the [Progress Document](/docs/README-progress.md) to see the current progress. -->

## Scope and Applications

> [!NOTE]
> This section is a summary generated from the [report](/docs/KTL2401_1155175983_1155174636_final_report_term2.pdf) by Grok. The contents have been double-checked by the author. 
> 
> Only this section covers the main content of the report and the remaining sections are about the details of setting up the project and the purpose of specific scripts within the repository.

This project aims to create a unified, semi-supervised contrastive-learning framework for hand gesture recognition. The framework is designed to adapt efficiently to various downstream tasks, such as human-computer interaction and sign language recognition, with minimal retraining or fine-tuning.

### Key Areas Explored

#### Static-Pose Representation Learning
- **Objective**: Map hand landmark inputs (shape $21 \times 3$) into feature embeddings (size $128$).
- **Approach**: Compared three encoder architectures:
  - Multi-layer Perceptron (MLP)
  - Graph Convolutional Network (GCN)
  - Graph Attention Network (GAT)
- **Hypotheses Tested**:
  1. Graph-based models (GCN and GAT), which leverage edge information, outperform MLP in accuracy and convergence speed. This was evaluated using supervised contrastive loss on the Lexset dataset.
  2. Incorporating a large unlabelled dataset (synthetic MANO data) with curriculum-based augmentations enhances model generalization.

#### Extension to Dynamic Gesture Recognition
- **Objective**: Extend the contrastive learning approach to recognize dynamic gestures.
- **Approach**: Utilize sequential architectures like Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) units to model temporal dependencies in gesture sequences.

### Results

While not fully achieved the original goals, our key findings include:

- **Static Gesture Recognition**:
  - Graph-based networks (e.g., GCN, GAT) are more effective, leveraging hand skeletal connections for improved accuracy and faster convergence.
  - Using large unlabelled datasets with curriculum learning enhances model generalization to new datasets and unseen gesture classes.
- **Dynamic Gesture Recognition**:
  - Hierarchical and part-wise architectures improve understanding of gesture structures.
  - Contrastive learning showed limited improvement over existing methods, indicating a need for more complex approaches.

### Future Work
- Develop a general hand gesture encoder capturing rotation- and scale-invariant features for rapid adaptation to tasks like dynamic gesture recognition.
- Investigate joint training of static and dynamic datasets using curriculum and contrastive learning to improve robustness.

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

### Architecture
- Dataset related: `scripts/datasets/`, `scripts/fake_data/`, `scripts/hand_only_supervised/`
- Models will be saved under the `runs` folder.

### Training Scripts (`training/`)

#### Contrastive Learning (`contrastive/`)
| Category | Script | Features |
|----------|---------|-----------|
| InfoNCE (Unsupervised) |
| | `train_unsup_gat.py` | ✓ InfoNCE<br>✗ Curriculum<br>✓ GAT/GCN variants |
| | `train_unsup_gat_curriculum_multi.py` | ✓ InfoNCE<br>✓ Curriculum |
| Mixed (Direct Augmentation) |
| | `train_supcon.py` | ✗ Curriculum |
| | `train_supcon_gat.py` | ✗ Curriculum |
| Mixed (Curriculum) |
| | `train_supcon_gat_curriculum.py` | ✓ Curriculum |
| | `train_supcon_gat_curriculum_multi.py` | ✓ Curriculum<br>✓ Multi-dataset support |

#### Temporal Models (`temporal/`)
| Category | Variant | Script | Key Features |
|----------|---------|--------|--------------|
| Ordinary LSTM | Base | `lstm_ordinary.py` | Simple sequence learning |
| | Hand Only | `lstm_ordinary_hand_only.py` | Hand keypoints only |
| | Hierarchy | `lstm_ordinary_hierarchy.py` | Hierarchical learning |
| Contrastive LSTM | Base | `lstm_contrastive_compressed.py` | Contrastive learning with compression |
| | Hierarchy | `lstm_contrastive_compressed_hierarchy.py` | Hierarchical contrastive learning |

All scripts can be run using: `python training/<category>/<script_name>.py`
Testing: Use `lstm_test.py` for all temporal models

