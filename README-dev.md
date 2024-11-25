# Major FYP

> Major FYP > Setup Guide

### Useful Links

[![GitHub](https://img.shields.io/badge/📄Description%20doc-purple?style=for-the-badge)](/README.md)
[![Demo & Setup Doc](https://img.shields.io/badge/📄%20Demo%20Doc-blueviolet?style=for-the-badge)](/demo/README.md)
[![](https://img.shields.io/badge/📄SETUP%20GUIDE-grey?style=for-the-badge)](/README-dev.md)
[![Reference Papers](https://img.shields.io/badge/📄Reference%20Papers-green?style=for-the-badge)](/README-references.md)
[![Planning Document](https://img.shields.io/badge/🔗%20Planning%20Document-blue?style=for-the-badge)](https://1drv.ms/w/s!Ago9nLnz9h82gosJqjlnjoOBK9SS-Q?e=YuU6Wy)
[![Midterm Report (Updating)](https://img.shields.io/badge/🔗%20Midterm%20Report-orange?style=for-the-badge)](/Major_FYP_Planning_Report%20(updating).pdf)


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

### Checkpoints

To download the checkpoints, use `curl` and place them under the `saved_models` directory. The following models are available:

* **EfficientNet+CELoss (30 epochs)**: [https://drive.google.com/drive/folders/128DE8fLQqMX3iL1_0a2KuMJtmAAFq4vs](https://drive.google.com/drive/folders/128DE8fLQqMX3iL1_0a2KuMJtmAAFq4vs)
* **EfficientNet+TripletLoss (Raw, 40 epochs)**: [https://drive.google.com/drive/folders/1Z9SMVnhLKA8j9L7I9SKLBeRl-damb9FE](https://drive.google.com/drive/folders/1Z9SMVnhLKA8j9L7I9SKLBeRl-damb9FE)

### Datasets

Put the datasets under `kaggle/input`.
Use curl to download the datasets.
Execute the code AT THE ROOT OF THIS PROJECT.

* **Synthetic-asl-dataset** 
    [![Source](https://img.shields.io/badge/Source-Kaggle-blue)](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet/data)<details>
    ```bash
    kaggle datasets download -d lexset/synthetic-asl-alphabet
    mkdir "kaggle/input/synthetic-asl-alphabet"
    tar -xf synthetic-asl-alphabet.zip -C kaggle/input/synthetic-asl-alphabet
    ```
    Search for installation guides for the kaggle command if it is not working.
    </details>

* **Roboflow-asl-alphabet-1**
    [![Source](https://img.shields.io/badge/Source-Kaggle-blue)](https://universe.roboflow.com/nmims-oawfg/sign-language-detectiom/dataset/1#)<details>
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
    </details>

* **Future...**

### Tests Conducted
Under the `tests` folder, you'll find the following experiments:

* **Preliminary Tests**: Clones of "Face Recognition with Siamese Network" [![Source](https://img.shields.io/badge/Source-Kaggle-blue)](https://www.kaggle.com/code/tatianakushniruk/face-recognition-with-siamese-network/notebook)
    * **Face Dataset Test (Local Execution)**: [![Face Dataset Test](https://img.shields.io/badge/Face%20Dataset%20Test-Local%20Execution-green)](/tests/1_siamese_face/face-recognition-with-siamese-network.ipynb)
        * **Location**: `tests/1_siamese_face/`
    * **Migration to ASL Dataset**: [![Migration to ASL Dataset](https://img.shields.io/badge/Migration%20to%20ASL%20Dataset-Test-green)](/tests/2_asl/)
        * **Location**: `tests/2_asl/`
* **EfficientNet+CELoss Tests**:
    * **Location**: `tests/3_modularization_test/cross_entropy_loss/`
    * **Training Test**: [![Training Test](https://img.shields.io/badge/Training%20Test-EfficientNet%2BCELoss-green)](/tests/3_modularization_test/cross_entropy_loss/modularization_train_ce.ipynb)
    * **Inference Test**: [![Inference Test](https://img.shields.io/badge/Inference%20Test-EfficientNet%2BCELoss-green)](/tests/3_modularization_test/cross_entropy_loss/modularization_test_inference.ipynb)
* **EfficientNet+TripletLoss (Raw) Tests**:
    * **Location**: `tests/3_modularization_test/triplet_loss_raw/`
    * **Training Test**: [![Training Test](https://img.shields.io/badge/Training%20Test-EfficientNet%2BTripletLoss%20(Raw)-green)](/tests/3_modularization_test/triplet_loss_raw/modularization_train_triplet.ipynb)
    * **Inference Test**: [![Inference Test](https://img.shields.io/badge/Inference%20Test-EfficientNet%2BTripletLoss%20(Raw)-green)](/tests/3_modularization_test/triplet_loss_raw/modularization_test_inference_triplet.ipynb)

### File Architecture

Every essential libraries are under `lib` directory.

* `data` loads from datasets.
* `models` specifies the architectures of the classifier.
* `trainers` stores the training script.
* `verifiers` stores the tools to evaluate the performance of the library.
* `config_loader`, `data_loader`, `model_loader` etc: the names are self-explanatory.