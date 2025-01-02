
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