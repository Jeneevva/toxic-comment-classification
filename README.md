# Toxic Comment Classification (NLP & Deep Learning)

## Project Overview
This project focuses on detecting toxic and abusive language in online comments using
Natural Language Processing (NLP) and Machine Learning techniques.

The goal is to automatically classify user-generated text into toxic and non-toxic categories,
including different types of toxicity such as hate speech, insults, threats, and harassment.

---

## Dataset
The project uses the **Jigsaw Toxic Comment Classification Dataset**, which is publicly available
on Kaggle. Due to file size limitations, the dataset is not included in this repository.
It can be downloaded from Kaggle:
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

- Source: Kaggle – Jigsaw Toxic Comment Classification Challenge
- Language: English
- Task Type: Multi-label text classification
- Number of samples: ~160,000 comments

Each comment may belong to one or more toxicity categories:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

---

## Project Structure
ml_project/
│
├── 01_preprocessing_eda.ipynb
├── 02_modeling.ipynb
├── README.md

---

## Methodology

### Data Preprocessing
- Removed missing and duplicate entries
- Cleaned text (lowercasing, removing punctuation and stopwords)
- Applied lemmatization
- Saved cleaned dataset for further analysis

### Exploratory Data Analysis (EDA)
- Analyzed class distribution
- Identified strong class imbalance
- Visualized toxicity label frequencies
- Confirmed multi-label nature of the task

### Model Development
Two models were implemented:

1. **Baseline Model**
   - TF-IDF vectorization
   - Logistic Regression (One-vs-Rest)

2. **Improved Model**
   - Tokenization and padding
   - LSTM-based deep learning model using Keras

---

## Evaluation
Models were evaluated using:
- Precision
- Recall
- F1-score (micro and macro averaged)

The baseline model performed well on frequent toxicity categories, while the LSTM model
demonstrated improved contextual understanding and better performance on minority classes.

---

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- NLTK
- Matplotlib, Seaborn

---

## Team
**Group:** SIS-2207  
**Team Members:**
- Tnasheva Zhanel  
- Alexeyeva Alina  
- Yerbolova Amina  

---

## Conclusion
This project demonstrates the effectiveness of machine learning and deep learning approaches
for toxic comment classification and highlights the importance of proper preprocessing,
EDA, and evaluation when working with imbalanced textual data.

