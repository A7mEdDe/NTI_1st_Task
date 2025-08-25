# -*- coding: utf-8 -*-
"""Flowers_CNN_Task.py"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import os


DATA_DIR = "data/flowers"

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

model = Sequential(
    [
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# تدريب النموذج
model.fit(train, epochs=10, validation_data=validation)

loss, accuracy = model.evaluate(validation)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🌸 Flowers Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded_file:
    # اقرأ الصورة
    img = tf.keras.utils.load_img(uploaded_file, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_labels = list(train.class_indices.keys())

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"**Prediction:** {class_labels[class_idx]}")
