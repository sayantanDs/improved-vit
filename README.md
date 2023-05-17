## Getting Started 

Clone or download this repository and install the pre-requisites:

- numpy
- matplotlib
- tensorflow
- tensorflow-io

```
pip install numpy matplotlib tensorflow tensorflow-io
```

### Preparing dataset:
Run the notebook ```create_mel_spectrograms.ipynb```. This will generate mel spectrograms of Google speech commands V2 dataset.

### Training model:
To train the proposed convolutional vision transformer model from scratch:
```
cd src
python train.py -m cvit
```

### Testing trained models:
```
cd src
python test.py -m "../trained_models/cvit/model.json" -w "../trained_models/cvit/cvit_100-0.952.h5"
```