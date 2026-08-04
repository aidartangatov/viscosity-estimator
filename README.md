# QuanNet

QuanNet is an open-source pipeline leveraging ResNet for predicting antibody solution viscosity. 

## Features:
 - **Open Source**: QuanNet is a completely open-source solution, making it readily accessible to both academic and 
                    commercial users. 
 - **ResNet Architecture**: Uses a deep neural network based on the ResNet architecture for viscosity predictions.
 - **Easy-to-Use CLI & Python API**: Seamlessly train, fine-tune, or infer directly from the command line or 
                                     within Python.

## Installation:

### Requirements:
- Docker installed on your machine.
- Python>=3.8 environment with pip.

### Installation Steps:
1. Build the docker image:
```commandline
docker build -t quannet .
```
2. Install the quannet package via pip:
```commandline
pip install .
```

## Usage

### Command-Line Interface (CLI)
QuanNet is equipped with a user-friendly CLI for easy interaction and can be accessed using the `quannet` command.

#### Prediction
Predict the viscosity of antibody solutions using a pretrained model:

```commandline
quannet predict model=<path-to-trained-model>.pt structures=datasets/full_dataset/full_dataset
```

#### Training
Train QuanNet on your dataset:

```commandline
quannet train dataset=datasets/full_dataset
```

### Python API:
For those who prefer using QuanNet within a Python environment, you can seamlessly integrate it into your scripts:

```python
from quannet import QuanNet

# Load a pretrained model
model = QuanNet("<path-to-trained-model>.pt")

# Predict viscosity
model.predict(structures="datasets/full_dataset/full_dataset")

# Train the model on your dataset
model.train(dataset="datasets/full_dataset", epochs=3)
```

### Contribute:
If you are interested in contributing to the development of QuanNet or have any questions,
feel free to open an issue or submit a pull request. We appreciate the community's support and contributions.

### Citation:
If you use QuanNet in your research, please cite our paper:

### License
MIT
