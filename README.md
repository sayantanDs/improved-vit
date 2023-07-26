# Introduction

This repository contains tensorflow implementation of improved ViT model for speech command recognition by combining convolutional features with the attention mechanism in transformer blocks.

# Getting Started 

Clone or download this repository and install the pre-requisites:

- numpy
- matplotlib
- tensorflow
- tensorflow-io

```
pip install numpy matplotlib tensorflow tensorflow-io
```

## Preparing dataset:
Run the notebook ```create_mel_spectrograms.ipynb```. This will generate mel spectrograms of Google speech commands V2 dataset.

## Training model:
To train the proposed convolutional vision transformer model from scratch:
```
cd src
python train.py -m cvit
```

## Testing trained models:
```
cd src
python test.py -m "../trained_models/cvit/model.json" -w "../trained_models/cvit/cvit_100-0.952.h5"
```
The improved ViT model trained on Google Sppech Commands V2 for 100 epochs is available at: ```./trained_models/cvit/cvit_100-0.952.h5```. It results in the follwing confusion matrix:

![cvit_confusion_matrix](https://github.com/sayantanDs/improved-vit/assets/39154403/27d95334-4ae5-409e-897a-d63e24092b9a)