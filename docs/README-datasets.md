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

## RWTH-PHOENIX-Weather 2014 T Dataset

[![Source](https://img.shields.io/badge/Source-Link-blue)](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
[![Source](https://img.shields.io/badge/Paper-CVPR-blue)](https://openaccess.thecvf.com/content_cvpr_2018/html/Camgoz_Neural_Sign_Language_CVPR_2018_paper.html)
![master](https://img.shields.io/badge/Updated-2018-yellow)
![](https://img.shields.io/badge/Unused-grey)

**Cite:** Necati Cihan Camgöz, Simon Hadfield, Oscar Koller, Hermann Ney, Richard Bowden, Neural Sign Language Translation, IEEE Conf. on Computer Vision and Pattern Recognition, Salt Lake City, UT, 2018.

**Path:** `data/raw/phoenix-2014-t`

**Installation:** Download the dataset from the link provided, and unzip it within the `data/raw` folder.

```bash
mkdir "data/raw/phoenix-2014-t"

# No progress bar
tar -xvf data/raw/phoenix-2014-T.v3.tar.gz -C data/raw/phoenix-2014-t --strip-components=1

# or WSL
pv data/raw/phoenix-2014-T.v3.tar.gz | tar -xf - -C data/raw/phoenix-2014-t --strip-components=1
```

## RWTH-PHOENIX-Weather 2014 MS Handshapes Dataset

[![Source](https://img.shields.io/badge/Source-Link-blue)](https://www-i6.informatik.rwth-aachen.de/~koller/1miohands-data/)
[![Paper](https://img.shields.io/badge/Paper-CVPR-blue)](https://www-i6.informatik.rwth-aachen.de/publications/download/1000/Koller-CVPR-2016.pdf)
![master](https://img.shields.io/badge/Updated-2016-yellow)
![](https://img.shields.io/badge/Unused-grey)

Largely unbalanced dataset

**Cite:** O. Koller, H. Ney, and R. Bowden. Deep Hand: How to Train a CNN on 1 Million Hand Images When Your Data Is Continuous and Weakly Labelled. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3793-3802, Las Vegas, NV, USA, June 2016.

**Path:** `data/raw/ph2014-handshape`

**Installation:** Download the dataset from the link provided, and unzip it within the `data/raw` folder.

```bash
# Test
mkdir "data/raw/ph2014-handshape/test"

# No progress bar
tar -xvf data/raw/ph2014-dev-set-handshape-annotations.tar.gz -C data/raw/ph2014-handshape/test --strip-components=1

# or WSL
pv data/raw/ph2014-dev-set-handshape-annotations.tar.gz | tar -xf - -C data/raw/ph2014-handshape/test --strip-components=1

# Replace `test` with `train` and the tar path with the train dataset path to unpack the train dataset.
```

