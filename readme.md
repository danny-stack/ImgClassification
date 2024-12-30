# CIFAR-10 Image Classification Project

A comprehensive comparison of different deep learning models on CIFAR-10 classification task using TensorFlow.

## Environment Setup
```bash
pip install tensorflow-gpu==2.10.0 protobuf==3.19.6 tensorflow-metadata==1.10.0 tensorflow_datasets==4.6.0 pillow numpy
```

## Usage
```bash
# Basic training
python train.py --model_name resnet18 --batch_size 128 --img_size 32

# With custom parameters
python train.py --model_name resnet18 \
                --batch_size 128 \
                --img_size 32 \
                --epochs 200 \
                --optimizer adamw \
                --initial_lr 0.001 \
                --lr_schedule exponential
```

## Parameters
- `--model_name`: Model architecture
- `--batch_size`: Training batch size
- `--img_size`: Input image size
- `--epochs`: Number of training epochs
- `--optimizer`: Optimizer (adamw/adam/sgd)
- `--initial_lr`: Initial learning rate
- `--lr_schedule`: Learning rate schedule (exponential/cosine)

## Results

| Model | Train Acc (%) | Val Acc (%) | Epochs | Initial LR |
|-------|--------------|-------------|---------|------------|
| ResNet18 | | | | |
| ResNet-Cifar | | | | |
| DenseNet | | | | |
| MobileNet | | | | |
| SqueezeNet | | | | |
| ViT | | | | |
| WideResNet | | | | |
| EfficientNet | | | | |

## Model Details
- ResNet: Basic and bottleneck blocks with skip connections
- DenseNet: Dense blocks with concatenative skip connections
- MobileNet: Lightweight architecture using depthwise separable convolutions
- SqueezeNet: Fire modules for parameter efficiency
- ViT: Vision Transformer with patch embedding
- WideResNet: Wider residual networks
- EfficientNet: Compound scaling of depth/width/resolution

## Data Augmentation
- Random horizontal flip
- Random brightness adjustment
- Random contrast adjustment

## Training Features
- Early stopping
- Learning rate scheduling
- Model checkpointing
- TensorBoard logging