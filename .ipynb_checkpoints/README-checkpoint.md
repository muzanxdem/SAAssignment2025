# Deep Learning with PyTorch: DL4IDS Workshop

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-black?style=for-the-badge&logo=weightsandbiases)](https://wandb.ai/site)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

Welcome to the **DL4IDS** repository! This workshop is designed to guide beginners through the fundamentals of Deep Learning using [PyTorch](https://pytorch.org/), eventually leading to a practical application in cybersecurity: an Intrusion Detection System (IDS).

## 📚 About This Workshop

This repository contains a series of Jupyter notebooks and a capstone project. It covers the essential building blocks of PyTorch, from basic tensor operations to computer vision, and demonstrates how to apply these concepts to real-world data.

### Who is this for?
- **Beginners** to Deep Learning and PyTorch.
- **Students/Researchers** interested in applying AI to cybersecurity (IDS).
- Anyone looking for a hands-on, code-first introduction to neural networks.

## 📂 Repository Structure

### 1. Learning Modules (Notebooks)
The core content is organized into sequential notebooks:

- **`00_pytorch_fundamentals.ipynb`**: Introduction to PyTorch tensors, operations, and basic syntax.
- **`01_pytorch_workflow.ipynb`**: The end-to-end PyTorch workflow: data loading, model building, training, and evaluation.
- **`02_pytorch_classification.ipynb`**: Building neural networks for binary and multi-class classification problems.
- **`03_pytorch_computer_vision.ipynb`**: Introduction to Convolutional Neural Networks (CNNs) and Computer Vision.
- **`04_pytorch_custom_datasets.ipynb`**: How to load and process custom datasets for your models.

Each module comes with corresponding exercises in the `exercises/` folder to test your understanding.

### 2. Capstone Project: Intrusion Detection System (IDS)
Located in the `IDS/` directory, this project applies the learned concepts to the **CICIDS2017** dataset.

- **Goal**: Build a deep learning model to detect network intrusions.
- **Key Files**:
  - `IDS_CICIDS2017.py`: Main training script.
  - `model.py`: Neural network architecture definition.
  - `preprocess.py`: Data preprocessing pipeline.
  - `config.yaml`: Configuration for training parameters.
  - `convert.py` & `onnxtest.py`: Utilities for converting models to ONNX format and testing.

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- CUDA 13.0+ (Optional)
- uv
- Basic knowledge of Python programming.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/afifhaziq/DL4IDS.git
   cd DL4IDS
   ```

2. **Set up the environment**
   This project uses `uv` for dependency management.

   If you have `uv` installed:
   ```bash
   uv sync
   ```

   Alternatively, you can install the dependencies manually using pip (refer to `pyproject.toml` for versions):
   ```bash
   pip install torch torchvision numpy matplotlib pandas scikit-learn jupyter wandb onnx onnxruntime tqdm
   ```

## 🛠️ Usage

### Running the Notebooks
To start learning, launch Jupyter Lab or Notebook:

```bash
jupyter lab
```
Open the notebooks in order (starting from `00_pytorch_fundamentals.ipynb`) and run the cells to follow along.

### Running the IDS Project
To train the IDS model:

1. Navigate to the `IDS` directory:
   ```bash
   cd IDS
   ```
2. Run the training script:
   ```bash
   python IDS_CICIDS2017.py
   ```
   *Note: Ensure you have the necessary [dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) downloaded and placed in the `data/CICIDS2017` directory.*

## 🤝 Acknowledgements
This workshop material is inspired by and adapted from the excellent [Learn PyTorch for Deep Learning](https://www.learnpytorch.io/) course by Daniel Bourke.

