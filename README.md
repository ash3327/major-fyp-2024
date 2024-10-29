# Major FYP
### Useful Links

[![Planning Document](https://img.shields.io/badge/Planning%20Document-blue?style=for-the-badge)](https://1drv.ms/w/s!Ago9nLnz9h82gosJqjlnjoOBK9SS-Q?e=YuU6Wy)
[![MIRO Board](https://img.shields.io/badge/MIRO%20Board-orange?style=for-the-badge)](https://miro.com/welcomeonboard/TVd0ejI4NzhYTHZJOTQ1NDhKSWtPUlFyUWZnaU9oYk15MzAxcnNCbUtNT1NRaTVQZENFUk5sSEJaVEJMZktGNXwzNDU4NzY0NTIxODcyMjM4MDQwfDI=?share_link_id=807166828631)

* PLEASE UPDATE YOUR PROGRESS IN THE PLANNING DOCUMENT.
* You can attach links to useful papers in the end of the planning document and please add comments properly so that everyone knows why the paper may be useful.

## Guides

### Environment Setup

```bash
# Requires: Python 3.10, CUDA 11.8, CUDNN 8 if on Windows
# Because tensorflow 2.10 (last supported GPU version without extra plugins) relies on Python 3.10.
virtualenv -p python3.10 venv

    # Windows
    .\venv\Scripts\activate 

    # Linux
    source venv/bin/activate

# Dependencies
pip install -r requirements.txt
```

### References
* Contrastive Learning: 
  * [Source Publication](https://www.researchgate.net/publication/347038642_Fisher_Discriminant_Triplet_and_Contrastive_Losses_for_Training_Siamese_Networks) 
  * [tensorflow: siamese network](https://www.kaggle.com/code/tatianakushniruk/face-recognition-with-siamese-network/notebook)
* Dataset:
  * [ASL Dataset by LEXSET@Kaggle](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet)

### Checkpoints:

Download these and put under `saved_models`.
Use curl to fetch the file.

* EfficientNet+CELoss, 30 epochs: https://drive.google.com/drive/folders/128DE8fLQqMX3iL1_0a2KuMJtmAAFq4vs

### Datasets:

Put the datasets under `kaggle/input`.
Use curl to download the datasets.
Execute the code AT THE ROOT OF THIS PROJECT.

1. Synthetic-asl-dataset ([Source](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet/data)):
    ```bash
    kaggle datasets download -d lexset/synthetic-asl-alphabet
    mkdir "kaggle/input/synthetic-asl-alphabet"
    tar -xf synthetic-asl-alphabet.zip -C kaggle/input/synthetic-asl-alphabet
    ```
    Search for installation guides for the kaggle command if it is not working.

2. Roboflow-asl-alphabet-1 ([Source](https://universe.roboflow.com/nmims-oawfg/sign-language-detectiom/dataset/1#)):
    ```bash
    # Windows
    curl -L "https://universe.roboflow.com/ds/CvkJnT8Is8?key=Gjdz88bXsh" > roboflow.zip
    mkdir "kaggle/input/roboflow-asl-alphabet-1"
    tar -xf roboflow.zip -C kaggle/input/roboflow-asl-alphabet-1
    del roboflow.zip

    # Linux
    rm roboflow.zip # after executing above commands.

    # License of usage: The API key is restricted to usage within the project.
    ```
    * Problem of this dataset: After close inspection, this dataset contains too much repeating images with slightly different augmentations. This dataset is basically unusable for training for this reason.

3. Future 

### Tests

Under `tests` folder.

* Clone of other's work for preliminary tests:
    * Clones of "Face Recognition with Siamese Network" ([Source](https://www.kaggle.com/code/tatianakushniruk/face-recognition-with-siamese-network/notebook))
        * Face Dataset test (local execution): `tests/1_siamese_face`
        * Migration to asl dataset [1]: `tests/2_asl`
* EfficientNet+CELoss:
    * Training test: `tests/3_modularization_test/modularization_test.ipynb`
    * Inference test: `tests/3_modularization_test/modularization_test_inference.ipynb`

### File Architecture

Every essential libraries are under `lib` directory.

* `data` loads from datasets.
* `models` specifies the architectures of the classifier.
* `trainers` stores the training script.
* `verifiers` stores the tools to evaluate the performance of the library.
* `config_loader`, `data_loader`, `model_loader` etc: the names are self-explanatory.

### To-Do



## What do we need to do now?

* Make the project goal and rationale clearer, specific up to the application and details.
* Preliminary analysis on multi-class classification on the ASL dataset, and possibly contrastive learning model on the ASL dataset. Refer to the section [Datasets](#datasets) for more information.

### Progress

* Triplet loss - Implement and study the effects of BATCH HARD NEGATIVES.
* Extraction of skeleton for hands, detecting multiple hands.
* YOLO-v8 for classification? (Benchmark)

* Searching for datasets: https://universe.roboflow.com/search?q=hand%2520gesture

### What will we deliver

* An interface (web or python) for capturing live video from user
* Keypoint extraction (which is easy)
* Connect to the model which we will be developing later, make sure that all models share the same format so that it is easy to integrate the modules later.
* Stage 1: Predicts the STATIC gesture (multi-class)
* Stage 2: Predicts the STATIC gesture (contrastive & cosine similarity)
    * Investigation on if it is possible to generalize a bit on unseen gestures.
    * Problem: requires larger dataset compared to the ASL (which should not be sufficient in theory).
    * Need to find larger dataset (TODO).
* Comparing benchmarks
* Change anything here if you find it bad.

### Checks

* We also need to confirm what devices do we need / are available.
* Personally Sam have two devices, one with RTX 4070 (laptop), one with RTX 3060Ti (desktop), possibly able to use the RTX 4070 + 4060 in the internship company with constraints.
* Kaggle available: T4 / P100 (30 hr / week, timeout 12 hr)

## Project Goal

The project goal is to develop a hand gesture recognition system.

Secondary goals if primary goal is achieved:
* Enable custom gesture addition without retraining the network.
* Match and annotate gestures from videos. 

## Significance of the Project (Rationale)

While most common gesture recognition models focuses on a fixed set of gestures, this project aims to improve the flexibility and usability of gesture recognition systems, with the following sub-goals:
* Address the need of custom gestures.
* Automate the annotation process for unannotated videos and allow developers to efficiently annotate certain gestures into the database.

Real Scenario of Application:

* Quick adaptation of the system to the controls of games, screen control (zooming, etc), sign language (gloss extraction based on similar gestures), etc with minimal re-training / finetuning if possible.
* Phase II (possibly) - Generative system for AR [Worldbox].

## Background

* Long-term motion characteristics [https://link.springer.com/article/10.1007/s13042-023-01987-3]
* Procedures to do:
    * Efficiency
    * Back tracking (attention / etc)
    * Review of methods
    * New method
      * Capturing based on skeleton (MediaPipe)
      * Combined channel method based on TwoStreamSLT
      * ? Distillation methods (just reducing size of model)
      * ? Boosting efficiency of networks
      * It is okay to fail to exceed their benchmark (afterall we are undergrads)

## Proposed Solutions

(just follow the pdf document)

## Proposed Timelines



## Requirements of the report

A report of 1 to 2 pages stating the Project Goal; Significance of the project (what, who and why); Problem  statement; Proposed solutions (deliverables); and Proposed timelines.This is the initial plan of the project. Changes to the initial plan is acceptable in the later stages whenever necessary.

## Difficulties

The difficulty of this project - which turned out to be too ambitious - includes the following:

1. Contrastive Learning
   * Current progress: Siamese Network + Triplet Loss
       * Dataset: [face dataset](https://www.kaggle.com/datasets/stoicstatic/face-recognition-dataset/)
       * Reference Implementation: [tensorflow: siamese network](https://www.kaggle.com/code/tatianakushniruk/face-recognition-with-siamese-network/notebook)
       * Progress: Execution successful on both Kaggle notebooks and local. Code will be updated later.<details>
         * ![alt text](/readme-src/image-1.png)
             * Trained without learning rate decay, so the performance is not as good (the face dataset consists of 1680 people and around 8-9 samples each)
         * ![alt text](/readme-src/image-2.png)
             * Trained with learning rate decay, extended patience, result is still not very satisfactory.
             * The similarity matrix (max cosine similarity): 
             * ![](/readme-src/output-confusion-max.png)
             * Min cosine similarity:
             * ![](/readme-src/output-confusion-min.png)
             * Direct comparison of the maximum inter-class cosine similarities and minimum intra-class cosine similarities:
             * ![](/readme-src/output-confusion-comparison.png)
             * We can observe that the current set does not work very well, as the diagonal values are NOT always the highest value within the group - that is, during recognition, SOME faces would still get misclassified.
             * But from the mean:
             * ![](/readme-src/output-confusion-mean.png)
             * we can see that in general most faces should get classfied correctly (further analsysis required).
         * T-SNE result on 15 distinct faces:
             * ![](/readme-src/output-tsne.png)
             * The clusters are clearly visible, indicating that they might still be easily separable.</details>
         * **Directions**: the original source did NOT generate new triplets for every iteration. This causes the triplets (specific anchor-positive-negative triplets) to be reused instead of fully utilizing the available images.
2. Time Series
   * Involves sequential data - dynamic gesture sequences. This can be complicated for starters, so we may first focus on static gestures.
3. Graph
   * It may be a viable and feasible option to utilize the **keypoints** captured from the gestures instead of the images/videos to train a classification / contrastive model. 
   * This way, we can perform more augmentations easily like offsetting the keypoints slightly, rotating the entire palm slightly, etc.. This is not possible on the raw image/video data.
   * Progress: Spatio-Temporal Graph Convolutional Neural Netowrk - (STGCN) with multi-class classification (during internship).
       * Preliminary analysis showed that it achieved around 90%+ accuracy trained just for a short amount of time **on a 20-class classification problem** for sign language.
       * Trained based on cropped and speed-adjusted gesture keypoint sequences.
       * ![alt text](/readme-src/image.png)
       * Mostly rely on my compnay senior's advise, so will need to reformulate and refactor the entire project in order to maintain a consistent style.
       * Need to solve: the accuracy instability issue.
       * The augmentations are not sufficient.
4. Adaptive Learning "on the fly"
   * This can basically be solved with contrastive learning.
5. Generalizing with unseen data with minimal retraining.
   * Suspect that this could be solved with contrastive learning.
   * Similar is done with this paper: 
       * E. Uboweja, D. Tian, Q. Wang, et al., On-device real-time custom hand gesture recognition, 2023. arXiv: 2309.10858 [cs.CV]. [Online]. Available: https://arxiv.org/abs/2309.10858.
       * In this paper, they trained a word-level fingerspelling model which utilizes a single-hand embedding sub-model to extract discriminative features as feature vectors - the weights are the transferred to a new model for training a custom gesture recognition model with minimal training data.
6. Small labelled dataset, with large unlabelled dataset.
    * This is not our first priority. 
    * This is a problem in the semi-supervised learning, and shares a lot of common characteristics with problems within contrastive learning.

Reference: video (temporal contrastive learning): https://arxiv.org/pdf/2101.07974

Contrastive good? https://arxiv.org/pdf/2011.13377 (tradeoffs) https://arxiv.org/pdf/2112.05340

## Datasets

### Static Datasets

* ASL alphabets [[go](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet)]

Goal: Comparison of performance of multiclass classification and contrastive learning.

### Dynamic Datasets

Goals: control the screen (touchless screen control)

* IPN Hand [[go](https://gibranbenitez.github.io/IPN_Hand/)]
<!-- * DHGD: Dynamic Hand Gesture Dataset for Skeleton-Based Gesture Recognition and Baseline Evaluations (Access via CUHK lib) [[go](https://ieeexplore-ieee-org.easyaccess1.lib.cuhk.edu.hk/document/10444226)] -->

* JESTER dataset [[go](https://www.kaggle.com/datasets/toxicmender/20bn-jester)]
* (Not accesible yet) DVS128 dataset [[go](https://coldpress.ai/product_page/1680201883079x468828458487031500)]

Goals: Generalizability over unseen gestures with minimal samples.

* E. Uboweja, D. Tian, Q. Wang, et al., On-device real-time custom hand gesture recognition, 2023. arXiv: 2309.
10858 [cs.CV]. [Online]. Available: https://arxiv.org/abs/2309.10858.
    * They trained a word-level fingerspelling model which utilizes a single-hand embedding sub-model to extract discriminative features as feature vectors - the weights are the transferred to a new model for training a custom gesture recognition model with minimal training data.

Goals: Generalizability over unseen gestures, including different body parts (including face, etc) - with minimal samples for retraining.

* HKSL reference (http://www.cslds.org/v4/)
    * Sign language investigation: https://www.sciencedirect.com/science/article/pii/S0024384121001601
    * Hard negatives & Phrasal separation (CANNOT (ability) vs CANNOT (permission))
    * Adapting to varying speeds of making the signs.
    * Names and context getting.

Goals: Sign language interpretation without the need to pretrain every gloss.

Goals: Extract extra information - like **distance**, **direction**, **object in contact**, etc. from the gesture.

## Techniques

* augments: angle (within 20 degrees difference)
* keypoints extraction

## Checkpoints

* Before 23 Sept: Multiclass on ASL, Constrastive on ASL.

## Plan

* Term 1: Gesture Recognition
* Term 2: (If Term 1 goes smoothly) Sign Langauge?