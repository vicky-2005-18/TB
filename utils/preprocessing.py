import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(dataset_dir, batch_size=32, target_size=(224, 224)):
    """
    Creates training and validation data generators with data augmentation.
    """
    
    # Validation split defines 80% train, 20% validation
    # Training set generator with Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 
    )

    # Validation generator should only rescale (normalize), NO augmentation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    print("Loading validation data...")
    val_generator = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator

def preprocess_single_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for prediction:
    - Load image
    - Resize to target_size (224x224)
    - Convert to array
    - Normalize values between 0-1
    - Add batch dimension
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0 # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension -> (1, 224, 224, 3)
    return img_array
