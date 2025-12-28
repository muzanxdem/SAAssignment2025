# IDS (Intrusion Detection System) Directory Documentation

## Overview

The `IDS/` directory contains a complete deep learning-based Intrusion Detection System implementation using PyTorch. The system is designed to classify network traffic or data samples into multiple categories (classes) using a feedforward neural network architecture. The implementation includes training, evaluation, model conversion to ONNX format, and ONNX runtime testing capabilities.

## Directory Structure

```
IDS/
├── IDS_CICIDS2017.py      # Main training and evaluation script
├── model.py                # Neural network model definition
├── preprocess.py           # Data preprocessing utilities
├── convert.py              # PyTorch to ONNX conversion script
├── onnxtest.py             # ONNX model testing and benchmarking
├── config.yaml             # Configuration file for training parameters
├── config-copy.yaml.bak    # Backup configuration (CICIDS2017 example)
├── scaler.pkl              # Saved StandardScaler for data normalization
├── IDS_CICIDS2017.pth      # Trained PyTorch model weights
├── IDS_CICIDS2017.onnx     # Exported ONNX model
├── IDS_CICIDS2017.onnx.data # ONNX model data file
└── wandb/                  # Weights & Biases experiment tracking logs
```

## Dependencies

### Required Python Packages

- **torch** - PyTorch deep learning framework
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning utilities (StandardScaler)
- **tqdm** - Progress bars
- **torchsummary** - Model architecture summary
- **wandb** - Experiment tracking and logging
- **pyyaml** - YAML configuration file parsing
- **onnxruntime** - ONNX model inference runtime

### Installation

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install dependencies (if not already installed)
pip install torch numpy scikit-learn tqdm torchsummary wandb pyyaml onnxruntime
```

## File Descriptions

### 1. `IDS_CICIDS2017.py` - Main Training Script

The primary script for training and evaluating the IDS model.

**Key Functions:**
- `train_model()` - Trains the model with early stopping and best model saving
- `evaluate_model()` - Evaluates model performance on validation/test sets
- `test_and_report()` - Generates comprehensive classification reports and confusion matrices

**Features:**
- Automatic device detection (CUDA/CPU)
- Early stopping with patience mechanism
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping for training stability
- Weights & Biases integration for experiment tracking
- Label range validation to prevent CUDA errors
- Automatic class count detection from data

**Usage:**
```bash
cd DL4IDS/IDS
python IDS_CICIDS2017.py
# or with uv
uv run IDS_CICIDS2017.py
```

**Expected Input Files:**
- `../data/CICIDS2017/train.npy` - Training data (features + labels)
- `../data/CICIDS2017/test.npy` - Test data
- `../data/CICIDS2017/val.npy` - Validation data
- `../data/CICIDS2017/class_names.npy` - Class names array
- `config.yaml` - Configuration file

**Output Files:**
- `IDS_CICIDS2017.pth` - Best model weights (saved during training)
- `scaler.pkl` - StandardScaler for data normalization
- WandB logs in `wandb/` directory

### 2. `model.py` - Neural Network Architecture

Defines the `IDSModel` class, a feedforward neural network for classification.

**Architecture:**
```
Input Layer (input_features)
    ↓
Linear(256) → BatchNorm1d → GELU → Dropout(0.2)
    ↓
Linear(128) → BatchNorm1d → GELU → Dropout(0.2)
    ↓
Output Layer (num_classes)
```

**Key Components:**
- **Input Features**: Configurable based on dataset
- **Hidden Layers**: 256 → 128 neurons
- **Activation**: GELU (Gaussian Error Linear Unit)
- **Regularization**: Batch Normalization + Dropout (20%)
- **Output**: Logits for multi-class classification

**Initialization:**
```python
model = IDSModel(input_features=num_features, num_classes=num_classes)
```

### 3. `preprocess.py` - Data Preprocessing

Handles data normalization and PyTorch DataLoader creation.

**Functions:**
- `preprocess()` - Main preprocessing for training
  - Fits StandardScaler on training data
  - Transforms train/test/val sets
  - Creates PyTorch DataLoaders
  - Saves scaler for later use

- `preprocess_onnx()` - Preprocessing for ONNX inference
  - Loads saved scaler
  - Transforms test data
  - Creates DataLoader for ONNX testing

**Data Format:**
- Input arrays: `(n_samples, n_features + 1)`
- Last column contains labels (integers)
- Features are normalized using StandardScaler (mean=0, std=1)

### 4. `convert.py` - ONNX Model Conversion

Converts trained PyTorch model to ONNX format for deployment.

**Features:**
- Loads trained model weights
- Exports to ONNX with dynamic batch size
- Supports both CPU and GPU inference

**Usage:**
```bash
python convert.py
```

**Requirements:**
- `IDS_CICIDS2017.pth` must exist (trained model)
- `config.yaml` must be configured
- Training data needed to determine input shape

**Output:**
- `IDS_CICIDS2017.onnx` - ONNX model file

**Note:** The script currently uses `config['class_names']` for num_classes, which may need updating if your dataset has a different number of classes than the config.

### 5. `onnxtest.py` - ONNX Model Testing

Tests ONNX model performance and benchmarks inference speed.

**Features:**
- Loads ONNX model with CUDA/CPU providers
- Performs inference on test set
- Calculates accuracy
- Measures inference time and throughput
- Warm-up run for accurate timing

**Usage:**
```bash
python onnxtest.py
```

**Output:**
- Accuracy percentage
- Total inference time
- Average time per batch
- Throughput (samples/second)

**Requirements:**
- `IDS_CICIDS2017.onnx` - ONNX model file
- `scaler.pkl` - Saved scaler
- `../data/CICIDS2017/test.npy` - Test data

### 6. `config.yaml` - Configuration File

YAML configuration file for training parameters.

**Current Configuration:**
```yaml
class_names:
   - 'neutral'
   - 'positive'
   - 'negative'

batch_size: 128
learning_rate: 0.01
num_epochs: 20
```

**Configuration Parameters:**
- `class_names`: List of class names (used for reporting, but actual classes are loaded from data)
- `batch_size`: Number of samples per batch
- `learning_rate`: Initial learning rate for AdamW optimizer
- `num_epochs`: Maximum number of training epochs

**Note:** The actual number of classes is automatically detected from `class_names.npy` file, so the `class_names` in config may not match the actual data classes.

## Workflow

### Complete Training Pipeline

1. **Data Preparation** (in `../data/CICIDS2017/`):
   ```bash
   cd ../data/CICIDS2017
   python preprocess_csv.py
   ```
   This generates:
   - `train.npy`, `test.npy`, `val.npy` - Data arrays
   - `class_names.npy` - Class names array

2. **Training**:
   ```bash
   cd ../../IDS
   python IDS_CICIDS2017.py
   ```
   - Loads data and configuration
   - Validates label ranges
   - Trains model with early stopping
   - Saves best model to `IDS_CICIDS2017.pth`

3. **Model Conversion** (Optional):
   ```bash
   python convert.py
   ```
   - Converts PyTorch model to ONNX format

4. **ONNX Testing** (Optional):
   ```bash
   python onnxtest.py
   ```
   - Tests ONNX model performance
   - Benchmarks inference speed

## Model Architecture Details

### Network Structure

```
Input (n_features)
    ↓
[Linear(256)] → [BatchNorm1d] → [GELU] → [Dropout(0.2)]
    ↓
[Linear(128)] → [BatchNorm1d] → [GELU] → [Dropout(0.2)]
    ↓
[Linear(num_classes)]
    ↓
Output (logits)
```

### Training Configuration

- **Optimizer**: AdamW
- **Loss Function**: CrossEntropyLoss
- **Learning Rate Scheduler**: ReduceLROnPlateau
  - Mode: min (monitor validation loss)
  - Factor: 0.1 (reduce by 10x)
  - Patience: 3 epochs
- **Early Stopping**: 
  - Patience: 5 epochs
  - Monitors: validation loss
- **Gradient Clipping**: max_norm=1.0
- **Regularization**: 
  - Dropout: 20%
  - Batch Normalization

## Data Format

### Input Data Structure

The numpy arrays (`train.npy`, `test.npy`, `val.npy`) should have shape:
```
(n_samples, n_features + 1)
```

- **Last column**: Integer labels (0 to num_classes-1)
- **Other columns**: Feature values (will be normalized)

### Label Requirements

- Labels must be integers in range `[0, num_classes-1]`
- The script validates label ranges before training
- Invalid ranges will raise a `ValueError`

### Class Names

- Stored in `class_names.npy` as a numpy array
- Automatically loaded during training
- Used for classification reports and confusion matrices

## Training Process

### Training Loop

1. **Data Loading**: Loads train/val/test arrays and class names
2. **Validation**: Checks label ranges match expected class count
3. **Preprocessing**: Normalizes features using StandardScaler
4. **Model Creation**: Initializes model with correct input/output dimensions
5. **Training**: 
   - Forward pass
   - Loss calculation
   - Backward pass with gradient clipping
   - Optimizer step
   - Learning rate scheduling
   - Early stopping check
6. **Evaluation**: Tests on test set with detailed metrics

### Monitoring

- **WandB Integration**: Logs training/validation loss, learning rate, accuracy
- **Console Output**: Progress bars, epoch summaries, final reports
- **Model Checkpointing**: Saves best model based on validation loss

## Evaluation Metrics

The `test_and_report()` function generates:

1. **Accuracy**: Overall classification accuracy
2. **Classification Report**: Per-class precision, recall, F1-score
3. **Confusion Matrix**: Class-wise prediction distribution

## Troubleshooting

### Common Issues

1. **ValueError: Object arrays cannot be loaded when allow_pickle=False**
   - **Solution**: Ensure all `np.load()` calls include `allow_pickle=True`
   - **Fixed in**: All scripts now use `allow_pickle=True`

2. **ValueError: could not convert string to float**
   - **Solution**: Ensure data preprocessing converts all non-numeric features to numeric
   - **Check**: Run `preprocess_csv.py` to regenerate data with proper encoding

3. **CUDA device-side assert triggered**
   - **Cause**: Label values out of range for model's output size
   - **Solution**: Script now validates label ranges and automatically detects num_classes from data
   - **Check**: Verify labels are in range `[0, num_classes-1]`

4. **ValueError: The least populated class has only 1 member**
   - **Cause**: Stratified split requires at least 2 samples per class
   - **Solution**: Script now handles this automatically by falling back to non-stratified split

5. **Model accuracy issues with many classes**
   - **Note**: With 69,490 classes (as in some datasets), classification is extremely challenging
   - **Consider**: Reducing classes, using different architectures, or data preprocessing

### Data Validation

Before training, the script validates:
- Label ranges match expected class count
- Data arrays are numeric (float64)
- Class names file exists and matches label count

## Performance Considerations

### Training
- **GPU Recommended**: Training is significantly faster on CUDA
- **Batch Size**: Adjust based on GPU memory (default: 128)
- **Early Stopping**: Prevents overfitting and saves time

### Inference
- **ONNX Runtime**: Faster inference than PyTorch
- **CUDA Provider**: Use GPU for faster ONNX inference
- **Batch Processing**: Process multiple samples simultaneously

## Weights & Biases Integration

The training script integrates with WandB for experiment tracking:

- **Project**: "Obi_DL4IDS"
- **Logged Metrics**: 
  - Train Loss
  - Validation Loss
  - Learning Rate
  - Accuracy
- **Model Watching**: Tracks gradients and parameters

To use WandB:
1. Install: `pip install wandb`
2. Login: `wandb login`
3. Runs are automatically logged to `wandb/` directory

## Notes

- The system automatically detects the number of classes from the data, not from config
- The `config.yaml` class_names may not match actual data classes
- All data must be numeric before training (use preprocessing script)
- Model saves best weights based on validation loss
- ONNX conversion requires the model to be in eval mode

## Future Improvements

Potential enhancements:
- Support for different model architectures
- Hyperparameter tuning capabilities
- Data augmentation techniques
- Multi-GPU training support
- Model quantization for deployment
- Real-time inference API

