import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf



# The primary EDL model for generating the saliency map 
model_edl = keras.models.Sequential()
model_edl.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model_edl.add(layers.BatchNormalization())
model_edl.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_edl.add(layers.BatchNormalization())
model_edl.add(layers.MaxPool2D((2, 2)))
model_edl.add(layers.Dropout(0.2))
model_edl.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_edl.add(layers.BatchNormalization())
model_edl.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_edl.add(layers.BatchNormalization())
model_edl.add(layers.MaxPool2D((2, 2)))
model_edl.add(layers.Dropout(0.3))
model_edl.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_edl.add(layers.BatchNormalization())
model_edl.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_edl.add(layers.BatchNormalization())
model_edl.add(layers.MaxPool2D((2, 2)))
model_edl.add(layers.Dropout(0.4))
model_edl.add(layers.Flatten())
model_edl.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
model_edl.add(layers.BatchNormalization())
model_edl.add(layers.Dropout(0.5))
model_edl.add(layers.Dense(10, activation='relu'))


# The secondary standard model for classify the saliency maps
model_secondary = keras.models.Sequential()
model_secondary.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 1)))
model_secondary.add(layers.BatchNormalization())
model_secondary.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_secondary.add(layers.BatchNormalization())
model_secondary.add(layers.MaxPool2D((2, 2)))
model_secondary.add(layers.Dropout(0.2))
model_secondary.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_secondary.add(layers.BatchNormalization())
model_secondary.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_secondary.add(layers.BatchNormalization())
model_secondary.add(layers.MaxPool2D((2, 2)))
model_secondary.add(layers.Dropout(0.3))
model_secondary.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_secondary.add(layers.BatchNormalization())
model_secondary.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_secondary.add(layers.BatchNormalization())
model_secondary.add(layers.MaxPool2D((2, 2)))
model_secondary.add(layers.Dropout(0.4))
model_secondary.add(layers.Flatten())
model_secondary.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
model_secondary.add(layers.BatchNormalization())
model_secondary.add(layers.Dropout(0.5))
model_secondary.add(layers.Dense(2, activation='softmax'))

model_regular = keras.models.Sequential()
model_regular.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model_regular.add(layers.BatchNormalization())
model_regular.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_regular.add(layers.BatchNormalization())
model_regular.add(layers.MaxPool2D((2, 2)))
model_regular.add(layers.Dropout(0.2))
model_regular.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_regular.add(layers.BatchNormalization())
model_regular.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_regular.add(layers.BatchNormalization())
model_regular.add(layers.MaxPool2D((2, 2)))
model_regular.add(layers.Dropout(0.3))
model_regular.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_regular.add(layers.BatchNormalization())
model_regular.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_regular.add(layers.BatchNormalization())
model_regular.add(layers.MaxPool2D((2, 2)))
model_regular.add(layers.Dropout(0.4))
model_regular.add(layers.Flatten())
model_regular.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
model_regular.add(layers.BatchNormalization())
model_regular.add(layers.Dropout(0.5))
model_regular.add(layers.Dense(10, activation='softmax'))