# Road-Network-Extraction-from-Satellite-Imagery
# U-Net Model for Image Segmentation

This repository contains code for training a U-Net model for image segmentation tasks, specifically for extracting road networks from satellite images. The model is implemented using TensorFlow and Keras.
Table of Contents

    Introduction
    Requirements
    Data Preparation
    Model Architecture
    Training the Model
    Results
    Usage
    Contributing
    License

# Introduction

The U-Net model is widely used for image segmentation tasks. It consists of an encoder-decoder architecture with skip connections to retain high-resolution features. This repository demonstrates how to build and train a U-Net model for road network extraction from satellite images.
Requirements

To run the code in this repository, you need to install the following dependencies:

    Python 3.x
    TensorFlow 2.x
    NumPy
    OpenCV
    scikit-learn

# You can install the required packages using pip:

```bash

pip install tensorflow numpy opencv-python scikit-learn
```
# Data Preparation

The dataset should contain satellite images and corresponding binary masks. The images should be named with a _sat.jpg suffix, and the masks should have a _mask.png suffix.

The folder structure should look like this:

```kotlin

data/
    Train1/
        image1_sat.jpg
        image1_mask.png
        image2_sat.jpg
        image2_mask.png
        ...
    Valid1/
        image1_sat.jpg
        image1_mask.png
        image2_sat.jpg
        image2_mask.png
        ...
```

# Model Architecture

The U-Net model is defined in the unet_model function. It consists of an encoder path with convolutions and max-pooling layers, a bottleneck, and a decoder path with upsampling and concatenation layers.
Training the Model

To train the model, run the provided script. The model will be compiled with the Adam optimizer and binary crossentropy loss. Early stopping and model checkpointing callbacks are used to monitor the training process.

```python

# Load training data
X_train, y_train = load_images_and_masks("Train1")

# Load validation data
X_valid, y_valid = load_images_and_masks("Valid1")

# Build the model
model = unet_model()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint('Suhora_Technology.h5', save_best_only=True)
]

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=50,
    batch_size=2,
    callbacks=callbacks
)

```
# Results

The model checkpoints and training history will be saved during training. The best model can be loaded using model.load_weights('Suhora_Technology.h5').
Usage

To use the trained model for predictions, load the model weights and pass the input images through the model.

```python

model.load_weights('Suhora_Technology.h5')
predictions = model.predict(Test)
```
