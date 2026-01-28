# Research on the Identification of Human and AI-Generated Images Using Convolutional Neural Networks

This repository contains code and experiments for identifying AI-generated images vs. human images using convolutional neural networks (CNNs). It includes dataset preparation scripts, multiple training variants, and basic prediction utilities.

## Contents
- Training scripts (various CNN/GPT variants): `AI_train*.py`, `AI_train_CNN*.py`, `AI_train_GPT*.py`, `VGG16.py`
- Dataset utilities: `create_dataset.py`, `AI_create_dataset1.py`, `AI_create_medium_dataset.py`, `output_resize_images.py`, `output_train_label.py`
- Prediction: `AI_predict.py`
- Misc utilities/experiments: `plot_train_test_loss.py`, `show_reshape_images.py`, `image_output.py`, `extract_jason.py`

## Requirements
- Python 3.8+
- TensorFlow / Keras
- NumPy
- OpenCV (if using image preprocessing scripts)

Install dependencies (example):

```bash
pip install tensorflow numpy opencv-python matplotlib
```

## Data
Large datasets are intentionally excluded from version control. If you have your own dataset:
1. Place images into the expected folder structure used by your chosen dataset script.
2. Run a dataset creation script to build the train/test splits and labels.

See the dataset scripts for the exact directory layout and expected image sizes.

## Training
Pick a training script and run it. Example:

```bash
python AI_train_CNN.py
```

Outputs (models, metrics, logs) are saved according to each script’s settings.

## Prediction
Example:

```bash
python AI_predict.py
```

## Notes
- Some scripts are experimental variants created during research iterations.
- The VGG16 weights file is included for convenience.

## License
Add your preferred license here.
