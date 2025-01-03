# Major FYP

> Major FYP > Datasets

### Useful Links
[![GitHub](https://img.shields.io/badge/📄Description%20doc-purple?style=for-the-badge)](/README.md)
[![Demo & Setup Doc](https://img.shields.io/badge/📄%20Demo%20Doc-blueviolet?style=for-the-badge)](/demo/README.md)
[![](https://img.shields.io/badge/📄TESTS%20SETUP%20GUIDE-yellow?style=for-the-badge)](/docs/README-dev.md)
[![](https://img.shields.io/badge/📄DATASETs-grey?style=for-the-badge)](/docs/README-datasets.md)
[![Reference Papers](https://img.shields.io/badge/📄Reference%20Papers-green?style=for-the-badge)](/docs/README-references.md)
[![Planning Document](https://img.shields.io/badge/🔗%20Planning%20Document-blue?style=for-the-badge)](https://1drv.ms/w/s!Ago9nLnz9h82gosJqjlnjoOBK9SS-Q?e=YuU6Wy)
[![First Term Report](https://img.shields.io/badge/🔗%20Term%201%20Report-orange?style=for-the-badge)](/docs/KTL2401_1155175983_1155174636_final_report.pdf)

## Lexset: Synthetic ASL Alphabet Dataset

[![Source](https://img.shields.io/badge/Source-Kaggle-blue)](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet/data)
![master](https://img.shields.io/badge/Updated-2021-yellow)
![](https://img.shields.io/badge/v1-green)

A static-image dataset containing 27000 images of dimensions 512 x 512 for alphabet finger-spelling in American Sign Language (ASL).

Consists of 27 classes, with 900 samples per class in training set, and 100 sample per class in the test set. One class is the BLANK class.

**Path**: `data/raw/synthetic-asl-alphabet/`

**Subfolders**: 
* train: `Train_Alphabet`
* test: `Test_Alphabet`
* folder structure within:
  * folder-ed by-class, independent images

**Potential Usage**: Training to capture the class features

<details open>
<summary>Installation Guide</summary>

```bash
kaggle datasets download -d lexset/synthetic-asl-alphabet
mkdir "data/raw/synthetic-asl-alphabet"
tar -xf synthetic-asl-alphabet.zip -C data/raw/synthetic-asl-alphabet
```
Search for installation guides for the kaggle command if it is not working.
</details>

## ICIP17 Stereo Hand Pose Dataset
[![Source](https://img.shields.io/badge/Source-GitHub-blue)](https://github.com/zhjwustc/icip17_stereo_hand_pose_dataset)
[![Source](https://img.shields.io/badge/Paper-Arxiv-blue)](https://arxiv.org/abs/1610.07214)
![master](https://img.shields.io/badge/Updated-2017-yellow)
![](https://img.shields.io/badge/Unused-grey)

Includes images and extracted keypoint sequence (shape: 3 x 21 x 1500), with no explicit label on classes.

**Path**: `data/raw/B2Counting`

**Potential Ussage**: May be suitable for capturing the essence of continuity

**Installation**: Clone the files from the link provided in the github repo.