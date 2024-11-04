# Retinal OCT Image Processing and Classification

This project implements a Super-Resolution GAN (SRGAN) for enhancing low-resolution retinal OCT (Optical Coherence Tomography) images, followed by classification of retinal conditions using the enhanced images.

## Dataset

The project uses retinal OCT images from [Kaggle's Retinal OCT Dataset](https://www.kaggle.com/code/paultimothymooney/detect-retina-damage-from-oct-images).

## Project Pipeline

### 1. Data Preparation

1. Download the original OCT dataset from Kaggle
2. Run the preprocessing scripts:
   ```bash
   python downscale.py   # Creates two downscaled datasets (128x128 and 32x32)
   python aggregate.py    # Combines train and validation images from 128x128 dataset
   ```

### 2. SRGAN Training and Inference

1. Configure paths in `config.py`:
   - Set the path for high-resolution images (128x128)
   - Specify output paths for model checkpoints
   - Configure other training parameters as needed

2. Train the SRGAN:
   ```bash
   python train.py
   ```
   This process:
   - Downscales input images to 32x32
   - Trains the SRGAN to generate 128x128 images
   - Saves discriminator and generator models at specified intervals

3. Generate enhanced images:
   ```bash
   python srgan_inf.py
   ```
   This script:
   - Uses the saved generator model
   - Creates 128x128 enhanced images from 32x32 input images
   - Saves the generated dataset for classification

### 3. Classification Models

Two classification models are trained and evaluated:

1. Model A (Baseline):
   ```bash
   python classify.py
   ```
   - Trains and tests on original downscaled 128x128 images

2. Model B (SRGAN-enhanced):
   ```bash
   python classify.py
   ```
   - Trains and tests on SRGAN-generated 128x128 images

## Project Structure

```
├── downscale.py         # Creates downscaled datasets
├── aggregate.py         # Combines training and validation data
├── config.py            # Configuration settings for SRGAN
├── train.py             # SRGAN training script
├── srgan.py             # SRGAN model
├── srgan_inf.py         # SRGAN inference script
└── classify.py          # Classification model training script
```

<!-- ## Requirements

(Add your project dependencies here)

## Results

(Add your model performance metrics and comparison between Model A and B here) -->