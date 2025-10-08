# edunet-fake-news-detector
The Fake News Detector is a machine learning-based system designed to classify news articles as fake (0) or real (1) based on their textual content. It uses natural language processing (NLP) techniques and multiple ML models to ensure high accuracy and reliability.
This version includes:

1. Project overview

2. Technologies used (frontend, backend, model, data)

3.Setup instructions

4. Workflow explanation

5. Deployment guide


 Fake News Detector using Machine Learning
 Overview

This project is a Fake News Detection System built using Machine Learning and Natural Language Processing (NLP).
It predicts whether a given news article or headline is Real or Fake, based on its textual content.

The system is trained on a labeled dataset of news articles and deployed using Streamlit to provide an interactive web-based interface.

 Project Architecture
flowchart TD
A[User Inputs News Text] --> B[Streamlit Frontend]
B --> C[Preprocessing & Stemming (NLTK)]
C --> D[TF-IDF Vectorization (Scikit-learn)]
D --> E[Logistic Regression Model]
E --> F[Prediction: Fake or Real]
F --> G[Output Shown on Web UI]

 Technologies Used
 Machine Learning Engine

Model Used: Logistic Regression

Library: scikit-learn

Purpose: Classify text as Fake (1) or Real (0)

Natural Language Processing (NLP)

Library: NLTK

Functions Used:

Stopword removal

Stemming (PorterStemmer)

Text cleaning using Regular Expressions (re)

Feature Extraction: TF-IDF Vectorization

Frontend (User Interface)

Framework: Streamlit

Role: Provides a simple and interactive web interface for users to input news text and view predictions instantly.

 Backend (Processing Engine)

Language: Python

Frameworks/Libraries: Streamlit + Scikit-learn

Role: Handles preprocessing, model inference, and communication between the frontend and ML model.

 Data Handling

Libraries: Pandas, NumPy

Dataset: train.csv (news dataset with labeled articles)

Purpose: Load, clean, and merge columns (author + title) into a single text feature (content).

 Folder Structure
fake-news-detector/
│
├── app.py                # Streamlit application (main code)
├── train.csv             # Dataset used for training
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation

 Dataset Details

File Name: train.csv

Columns:

id: Unique identifier

title: News headline

author: News author

text: Full news article text

label: 0 = Real News, 1 = Fake News

Shape: 20,800 rows × 5 columns

 Model Workflow
Step	Description	Tools Used
1️⃣	Load dataset and handle missing values	Pandas
2️⃣	Combine text fields (author + title)	Python
3️⃣	Clean text using regex	re
4️⃣	Remove stopwords & apply stemming	NLTK
5️⃣	Convert text into TF-IDF vectors	Scikit-learn
6️⃣	Split data (80-20) for training/testing	Scikit-learn
7️⃣	Train Logistic Regression model	Scikit-learn
8️⃣	Evaluate model performance (accuracy)	Scikit-learn
9️⃣	Build Streamlit UI for prediction	Streamlit
 Model Performance
Metric	Accuracy
Training Accuracy	98.6%
Testing Accuracy	97.6%



