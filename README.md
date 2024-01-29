# Neural Artisan: MINIST Digit Recognition with Deep Learning Precision

## Overview

Neural Artisan is a project that employs deep learning precision to recognize hand-written digits from the MNIST dataset. The implementation utilizes TensorFlow, a powerful machine learning framework, to create a Convolutional Neural Network (CNN). The model is trained to accurately classify digits from 0 to 9, achieving impressive accuracy on both training and validation sets.

## Key Features

- **Data Import and Preprocessing:** The project starts by importing necessary libraries and loading the MNIST dataset. Basic Explorative Data Analysis (EDA) is performed to understand the data, and digits are visualized to provide insights.

- **Data Normalization and Splitting:** The dataset is normalized to a range of [0, 1]. It is then split into training and validation sets for model training.

- **Convolutional Neural Network:** A CNN is constructed using TensorFlow's Keras API. The model consists of convolutional and pooling layers, followed by fully connected layers. The network is compiled with the Adam optimizer and sparse categorical crossentropy loss.

- **Model Training:** The CNN is trained on the training set over 40 epochs, showing the training and validation accuracy over each epoch.

- **Softmax Activation and Prediction:** The softmax activation is applied to the model's predictions for both training and validation sets. The predictions are converted to categorical labels using argmax.

- **Error Calculation:** The project calculates the training and validation errors by comparing predicted labels with actual labels.

- **Results:** The trained model achieves high accuracy on the validation set, and the predictions on the test set are saved in a submission file.

- **Visualization:** The accuracy and validation accuracy are visualized over epochs. Randomly selected digits from the validation set are also plotted, showing actual and predicted labels.
- 

## Project Structure

- **Neural_Artisan.ipynb:** Jupyter Notebook containing the main project code, including data loading, preprocessing, model building, training, evaluation, and result visualization.

- **mnist-dataset :** [MNIST dataset on Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) (Note: The dataset is not directly included in this repository. You can download it from Kaggle using the provided link.)

- **submission.csv:** CSV file containing the predicted labels for the test set.

- **requirements.txt:** File listing the project dependencies, including TensorFlow and other necessary libraries.
  
## How to Use

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
2.**Install the required dependencies:**

```bash
   pip install -r requirements.txt
