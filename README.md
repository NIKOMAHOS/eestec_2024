# Sentiment Analysis in Twitter Posts - EESTech Challenge 2024

## Project Overview

This project was developed as part of the EESTech Challenge 2024 Machine Learning competition. The aim was to classify the sentiment of tweets (positive or negative) using supervised learning and explore clustering techniques for exploratory data analysis.

## Problem Statement

The task involves:
1. Developing a **supervised learning model** to classify tweets into sentiment categories.
2. Implementing **unsupervised learning methods** to explore clusters in the dataset.

## Dataset

We used the [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140/data), which contains 1.6 million tweets annotated with sentiment labels:
- **Positive**: Labeled as 0
- **Negative**: Labeled as 1

From this dataset, 200,000 tweets were randomly selected (100,000 for each class) for training and testing.

## Approach

### 1. Data Preprocessing
- Removed unwanted data elements, including mentions, URLs, special characters, and duplicates.
- Split the dataset into:
  - **Training Data**: 80% of the dataset.
  - **Testing Data**: 20% of the dataset.
- Vectorized tweets for machine learning models.

### 2. Supervised Learning
- Built a **Feed-Forward Neural Network** using a Multilayer Perceptron (MLP) architecture:
  - 4 hidden layers.
  - Batch normalization and dropout for regularization.
  - ReLU activation in hidden layers; sigmoid activation in the output layer.
- Evaluated the model using metrics like **accuracy** and confusion matrices.

### 3. Unsupervised Learning
- Applied **K-Means Clustering** to group tweets based on mean sentiment values.
- Evaluated clusters using the **Davies-Bouldin Index**.

## File Structure

- `final.ipynb`: Jupyter Notebook containing all implementation details, including preprocessing, model training, and evaluation.
- *Report*: A detailed explanation of the architecture, algorithms, and methodology used. (See attached PDF and DOCX files for more details.)

## How to Run

1. Clone this repository:
   git clone https://github.com/your-repo-url.git
   
2. Install Dependencies:
   pip install -r requirements.txt

 3. Open and run the `final.ipynb` notebook:
   jupyter notebook final.ipynb
