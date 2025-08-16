# Image-Classification-using-Tensorflow

*COMPANY*:CODTECH IT SOLUTIONS

*NAME*:RAGHA SREYA

*INTERN ID*:CT04DZ771

*DOMAIN*: DATA SCIENCE

*DURATION*:4 WEEKS

*MENTOR*:NEELA SANTOSH

##Project Overview

The primary goal of this project is to develop an automated image classification system using Deep Learning techniques, specifically Convolutional Neural Networks (CNNs), implemented in TensorFlow with Keras. Image classification is a fundamental computer vision task that involves assigning labels or categories to images based on their visual content. The model in this project is trained to classify images into one of ten predefined categories using the CIFAR-10 dataset, which is a standard benchmark dataset in the field of machine learning and computer vision.

The CIFAR-10 dataset contains 60,000 color images of size 32x32 pixels, equally distributed across 10 distinct classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. It is split into 50,000 training images and 10,000 testing images, making it suitable for evaluating deep learning models.

This project successfully demonstrates the process of building and training an image classification model using TensorFlow and Keras in google colab. The CNN-based architecture proves effective in extracting meaningful features from small, low-resolution images, and the methodology can be scaled or adapted to other datasets and real-world applications. The trained model provides a foundation for more complex image recognition systems, which can incorporate transfer learning, deeper architectures, and larger datasets for improved accuracy and robustness.

The CNN model is capable of learning meaningful visual patterns from the CIFAR-10 dataset, achieving competitive accuracy levels compared to baseline approaches. Visualization of accuracy and loss trends provides insights into the model’s learning behavior. The results demonstrate that CNNs, even with a relatively simple architecture, can achieve strong performance on image classification tasks.

Methodology

1. Data Preprocessing
-The CIFAR-10 dataset is loaded using TensorFlow’s built-in datasets API.
-The pixel values of the images are normalized from the range 0–255 to 0–1 for faster convergence during training.
-The dataset is split into training and testing sets to evaluate generalization performance.
2. Model Architecture
-The CNN model is constructed using the Sequential API in Keras, with the following layers:
-Conv2D layers: Extract spatial features from the input images using convolution operations.
-MaxPooling2D layers: Reduce spatial dimensions and computational complexity while retaining important features.
-Flatten layer: Converts 2D feature maps into a 1D vector for dense layers.
-Dense layers: Fully connected layers to combine extracted features for classification.
-The final dense layer outputs 10 values, corresponding to the number of classes, without an activation function (logits), as softmax activation is applied in the loss function.
3. Model Compilatio-Optimizer: Adam optimizer for efficient gradient descent.
-Loss Function: Sparse Categorical Crossentropy (with from_logits=True) for multi-class classification.
-Metrics: Accuracy to track model performance during training and testing.
4. Model Training
-The model is trained for 10 epochs, with validation data provided for monitoring overfitting.
-Batch size is chosen to balance training speed and memory usage.
5. Evaluation and Visualization
-After training, the model is evaluated on the test dataset to determine its accuracy.
-Accuracy and loss curves are plotted for both training and validation sets to visualize learning trends and detect potential overfitting or underfitting.
