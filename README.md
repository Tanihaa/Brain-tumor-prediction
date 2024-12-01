Brain Tumor Detection Tool  

This project is a machine learning-based tool designed to detect brain tumors in MRI images. It leverages two distinct neural network architectures—a Convolutional Neural Network (CNN) and an Artificial Neural Network (ANN)—to provide predictions. Users can upload an MRI image through an interactive Gradio interface, and the tool displays predictions from both models.  

Features  
- Dual Model Prediction: Combines CNN and ANN models to analyze MRI images.  
- Interactive Interface: Built using Gradio for easy user interaction.  
- Efficient and Scalable: Handles MRI image classification with high accuracy using TensorFlow.  

Machine Learning Models  

Convolutional Neural Network (CNN)  
- Architecture:  
  - Input: 200x200x3 MRI images.  
  - Layers:  
    - 3 convolutional layers with ReLU activation and max-pooling.  
    - Fully connected layer with 128 neurons and a dropout rate of 0.5.  
    - Sigmoid output layer for binary classification (tumor vs. no tumor).  
- Optimizer: Adam.  
- Loss Function: Binary Crossentropy.  
- Purpose: Ideal for spatial feature extraction in images.  

Artificial Neural Network (ANN)  
Architecture:  
  Input: Flattened 200x200x3 MRI images.  
  Layers:  
    - Fully connected dense layers with 512 and 256 neurons, each followed by dropout layers (0.5).  
    - Sigmoid output layer for binary classification.  
  Optimizer: Adam.  
  Loss Function: Binary Crossentropy.  
  Purpose: A simpler model for generalized image classification.  

Technologies Used  
TensorFlow/Keras: For building and training the CNN and ANN models.  
Gradio: For creating the user-friendly web interface.  
NumPy: For efficient data manipulation.  


