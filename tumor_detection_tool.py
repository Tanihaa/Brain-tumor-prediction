import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import gradio as gr

def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_ann_model():
    model = Sequential([
        Flatten(input_shape=(200, 200, 3)),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = create_cnn_model()
ann_model = create_ann_model()

def predict_image(image):
    image = image.resize((200, 200))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    cnn_prediction = cnn_model.predict(image_array)
    cnn_result = "Brain Tumor Detected" if cnn_prediction[0][0] > 0.5 else "No Brain Tumor Detected"

    ann_prediction = ann_model.predict(image_array)
    ann_result = "Brain Tumor Detected" if ann_prediction[0][0] > 0.5 else "No Brain Tumor Detected"

    return cnn_result, ann_result

gr_interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs=[
        gr.Textbox(label="CNN Model Prediction"),
        gr.Textbox(label="ANN Model Prediction")
    ],
    title="Brain Tumor Detection",
    description="Upload an MRI image to detect if there is a brain tumor present. The predictions from both CNN and ANN models will be displayed in separate boxes."
)

gr_interface.launch(). give a short description for this project
