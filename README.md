# Major FYP

## Objectives

Requires immediate clarification.

## Requirements of the report

A report of 1 to 2 pages stating the Project Goal; Significance of the project (what, who and why); Problem  statement; Proposed solutions (deliverables); and Proposed timelines.This is the initial plan of the project. Changes to the initial plan is acceptable in the later stages whenever necessary.

## Difficulties

The difficulty of this project - which turned out to be too ambitious - includes the following:

1. Contrastive Learning
   * Current progress: Siamese Network + Triplet Loss
       * Dataset: [face dataset](https://www.kaggle.com/datasets/stoicstatic/face-recognition-dataset/)
       * Reference Implementation: [tensorflow: siamese network](https://www.kaggle.com/code/tatianakushniruk/face-recognition-with-siamese-network/notebook)
       * Progress: Execution successful on both Kaggle notebooks and local. Code will be updated later.
2. Time Series
   * Involves sequential data - dynamic gesture sequences. This can be complicated for starters, so we may first focus on static gestures.
3. Graph
   * It may be a viable and feasible option to utilize the **keypoints** captured from the gestures instead of the images/videos to train a classification / contrastive model. 
   * This way, we can perform more augmentations easily like offsetting the keypoints slightly, rotating the entire palm slightly, etc.. This is not possible on the raw image/video data.
   * Progress: Spatio-Temporal Graph Convolutional Neural Netowrk - (STGCN) with multi-class classification (during internship).
       * Preliminary analysis showed that it achieved around 90%+ accuracy trained just for a short amount of time.
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

## Datasets

### Static Datasets

* ASL alphabets [[go](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)] [[also this](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet)]

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
