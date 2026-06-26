# 🏥 Clinical Trial Disease Category Classification
### Using NLP and Machine Learning

---

## 📋 Project Overview

This project classifies clinical trial summaries into different disease categories using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The application preprocesses clinical trial text, extracts features using TF-IDF, trains multiple machine learning models, and predicts the disease category through an interactive Streamlit web application.

**Domain:** Healthcare Analytics | Medical NLP

---

## 🎯 Objectives

- Preprocess clinical trial text data
- Apply NLP techniques (Tokenization, Lemmatization, TF-IDF)
- Train multiple Machine Learning classification models
- Evaluate model performance
- Build an interactive Streamlit web application
- Predict disease categories from clinical trial summaries

---

# 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| NLP | NLTK, TF-IDF |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web Framework | Streamlit |

---

# 🤖 Machine Learning Models

- Logistic Regression ✅ *(Best Model - 85% Accuracy)*
- Random Forest
- Naive Bayes
- Support Vector Machine (SVM)

---

# 📁 Project Structure

```text
Clinical-Trial-Disease-Category-Classification/
│
├── clinical_trial_classification.py      # Model training script
├── app.py                               # Streamlit web application
├── requirements.txt                     # Required libraries
│
├── models/
│   ├── disease_classification_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── visualizations/
│   ├── confusion_matrix.png
│   ├── accuracy_comparison.png
│   └── wordcloud.png
│
└── README.md
```

---

# 💻 Installation

## Clone the Repository

```bash
git clone https://github.com/yourusername/Clinical-Trial-Disease-Category-Classification.git
```

```bash
cd Clinical-Trial-Disease-Category-Classification
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

## Train the Model

```bash
python clinical_trial_classification.py
```

---

## Run the Streamlit Application

```bash
streamlit run app.py
```
---

# 📊 Model Performance

| Model | Accuracy | F1 Score |
|--------|----------|-----------|
| Logistic Regression | **85.0%** | **84.0%** |
| Random Forest | 82.0% | 81.0% |
| Naive Bayes | 79.0% | 78.0% |
| Support Vector Machine (SVM) | 84.0% | 83.0% |

### 🏆 Best Model

**Logistic Regression**

- Accuracy: **85%**
- F1 Score: **84%**

---

# 🔬 Methodology

### 1️⃣ Data Collection
- Load clinical trial dataset

### 2️⃣ Data Preprocessing
- Remove punctuation
- Convert text to lowercase
- Tokenization
- Stopword removal
- Lemmatization

### 3️⃣ Feature Extraction
- TF-IDF Vectorization

### 4️⃣ Model Training
- Logistic Regression
- Random Forest
- Naive Bayes
- Support Vector Machine

### 5️⃣ Model Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### 6️⃣ Deployment
- Streamlit Web Application

---

# 📱 Streamlit Web App Features

✅ Disease Category Prediction

✅ Dataset Overview

✅ Model Performance Visualization

✅ Interactive Charts

✅ Batch Prediction using CSV Upload

---

# 📦 Requirements

```
pandas
numpy
nltk
scikit-learn
matplotlib
seaborn
plotly
streamlit
wordcloud
```

---

# 📈 Visualizations

- Disease Category Distribution
- Accuracy Comparison
- Confusion Matrix
- Word Cloud
- Feature Importance
- Classification Report

---

# 📝 Project Deliverables

- ✅ Data Preprocessing Pipeline
- ✅ NLP Feature Engineering
- ✅ Multiple ML Classification Models
- ✅ Trained Model Files (.pkl)
- ✅ Streamlit Web Application
- ✅ Performance Evaluation
- ✅ Interactive Visualizations
- ✅ Complete Documentation

---

# 📚 Future Improvements

- Deep Learning Models (LSTM/BERT)
- Hyperparameter Optimization
- Clinical Trial Recommendation System
- API Deployment using FastAPI
- Docker Containerization
- Cloud Deployment (AWS/Azure)



