# Major FYP

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

## Objectives

