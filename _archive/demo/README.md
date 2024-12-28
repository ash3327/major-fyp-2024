# Major FYP

> Major FYP > Demo Document

### Useful Links
[![GitHub](https://img.shields.io/badge/📄Description%20doc-purple?style=for-the-badge)](/README.md)
[![Demo & Setup Doc](https://img.shields.io/badge/📄%20Demo%20Doc-grey?style=for-the-badge)](/tests/contrastive/README.md)
[![](https://img.shields.io/badge/📄TESTS%20SETUP%20GUIDE-yellow?style=for-the-badge)](/docs/README-dev.md)
[![Reference Papers](https://img.shields.io/badge/📄Reference%20Papers-green?style=for-the-badge)](/docs/README-references.md)
[![Planning Document](https://img.shields.io/badge/🔗%20Planning%20Document-blue?style=for-the-badge)](https://1drv.ms/w/s!Ago9nLnz9h82gosJqjlnjoOBK9SS-Q?e=YuU6Wy)
[![First Term Report](https://img.shields.io/badge/🔗%20Term%201%20Report-orange?style=for-the-badge)](/docs/KTL2401_1155175983_1155174636_final_report.pdf)

## Setting Up & Usage

* `cd` to `demo/`.
* Run `app.py`.
* Go to `http://127.0.0.1:5000/` to view the results.

## Data

* `python to_landmark_csv_augmented.py` to generate the augmented dataset.
* Result stored in `augmented_hand_landmarks.csv`.

## Training

* `python ce_augmented_deep.ipynb` to train the model.
* Class means stored in `class_means.json`.
* Model weights stored in `embedding_model.h5`.

## Inference

* `python ce_augmented_test.ipynb` to visualize.
* **Only ce_augmented_test_deep.ipynb works.**