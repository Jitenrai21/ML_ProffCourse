import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input, MaxPooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers, models


# Function to build the first custom CNN model
def build_custom_model():
    img_height = 256
    img_width = 256
    data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_height,img_width,3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
    )
    # Input layer
    input_layer = Input(shape=(256, 256, 3))


    model = Sequential()
    model.add(data_augmentation)
    model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))


    model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))


    model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))


    model.add(Flatten())


    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
   
    # Load the weights for this model
    model.load_weights(r"C:\Users\ACER\gitClones\ML_ProffCourseModels\models\dog_vs_cat.keras")  # Adjust the path
    return model


# Function to build the InceptionV3 model
def build_inception_model():
    # Input layer
    # input_layer = Input(shape=(200, 200, 3))


    # Load the InceptionV3 base model with pre-trained ImageNet weights
    base_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(256, 256, 3)
    )


    base_model.trainable = False


    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])


   # Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


    # Load the trained weights
    model.load_weights(r"C:\Users\ACER\gitClones\ML_ProffCourseModels\models\incep_dog_cat_classifier.h5")
    return model