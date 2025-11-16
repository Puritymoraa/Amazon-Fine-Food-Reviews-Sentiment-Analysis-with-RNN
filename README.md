# Amazon-Fine-Food-Reviews-Sentiment-Analysis-with-RNN
A deep learning project that uses Recurrent Neural Networks (RNN) with and without attention mechanisms to classify Amazon Fine Food Reviews as positive or negative.

This project implements sentiment analysis on the Amazon Fine Food Reviews dataset using deep learning. I built two models:

Baseline RNN - A standard LSTM-based recurrent neural network
RNN with Attention - An enhanced version that uses attention mechanism to focus on important words

The attention mechanism not only improves model performance but also provides interpretability by showing which words the model focuses on when making predictions.
🎯 Objectives

Develop practical understanding of RNNs and attention mechanisms
Apply RNN to real-world NLP problem following best practices
Train and evaluate neural networks for sentiment classification
Compare model performance with and without attention
Visualize attention weights to understand model decisions

Dataset
Source: Amazon Fine Food Reviews on Kaggle
Details:

Total reviews: 568,454
Features: Product ID, User ID, Score, Summary, Review Text
Time span: October 1999 - October 2012

Preprocessing:

Ratings 4-5 classified as Positive (label = 1)
Ratings 1-2 classified as Negative (label = 0)
Rating 3 (neutral) excluded from analysis
Final dataset balanced with equal positive and negative samples

Technologies Used

Python 3.x
PyTorch - Deep learning framework
NumPy - Numerical computing
Pandas - Data manipulation
Matplotlib & Seaborn - Data visualization
Scikit-learn - Data splitting and metrics
Google Colab - Training environment (GPU acceleration)

Methodology
1. Data Preprocessing

Text Cleaning: Lowercase conversion, HTML tag removal, special character handling
Class Balancing: Undersampling to handle class imbalance
Train-Val-Test Split: 70%-15%-15% stratified split
Tokenization: Building vocabulary of 10,000 most common words
Sequence Padding: Fixed length of 200 tokens

2. Model Architecture
Baseline RNN Model
Input (Sequence) → Embedding Layer → LSTM Layers → Fully Connected → Output (Sentiment)
Hyperparameters:

Embedding dimension: 100
Hidden dimension: 128
Number of LSTM layers: 2
Dropout: 0.3
Batch size: 64
Learning rate: 0.001
Optimizer: Adam

RNN with Attention
Input → Embedding → LSTM → Attention Mechanism → Fully Connected → Output
The attention layer computes importance scores for each word and creates a context vector by weighted sum of LSTM outputs.
3. Training Process

Loss Function: Binary Cross Entropy
Optimization: Adam optimizer
Epochs: 10 (with early stopping)
Hardware: GPU acceleration via Google Colab
Validation: Model selection based on validation loss

4. Evaluation Metrics

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
