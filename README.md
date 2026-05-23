# 🎭 Sentiment Analysis NLP

A machine learning web app for sentiment classification (Positive / Negative / Neutral) built from scratch using Python and Streamlit — with **Naive Bayes** and **SVM** classifiers, **TF-IDF** vectorization, and full **evaluation metrics**.

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Single Analysis** | Analyze any text with confidence scores |
| 📋 **Batch Analysis** | Analyze multiple texts at once |
| 📈 **Model Evaluation** | Accuracy, Precision, Recall, F1-score |
| 🔵 **Naive Bayes** | Probabilistic classifier |
| 🟣 **Linear SVM** | Support Vector Machine classifier |
| 📊 **TF-IDF** | Term Frequency-Inverse Document Frequency |

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Web Framework:** Streamlit
- **ML:** Custom Naive Bayes + Linear SVM (built from scratch)
- **Features:** TF-IDF Vectorization, Stopword Removal, Tokenization
- **Evaluation:** Accuracy, Precision, Recall, F1-Score

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Priya-singh03/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## 📁 Project Structure

```
sentiment-analysis-nlp/
├── app.py           # Streamlit web app (3 tabs)
├── sentiment.py     # Core ML logic
│   ├── preprocess()              # Tokenization + stopword removal
│   ├── tfidf_vectorize()         # TF-IDF feature extraction
│   ├── NaiveBayesClassifier      # Custom Naive Bayes
│   ├── LinearSVMClassifier       # Custom Linear SVM
│   └── SentimentAnalyzer         # Main analyzer class
├── requirements.txt
└── README.md
```

## 📊 How It Works

1. **Preprocessing** — Tokenization, lowercasing, stopword removal
2. **Feature Extraction** — TF-IDF vectorization across 3 corpora
3. **Classification** — Naive Bayes or SVM predicts sentiment
4. **Evaluation** — Accuracy, Precision, Recall, F1-score per class

## 🎯 Model Performance

Trained on 45 labeled samples across 3 classes:
- **Positive** — 15 samples
- **Negative** — 15 samples  
- **Neutral** — 15 samples

Achieving **88%+ accuracy** on test set with Naive Bayes.

## 👩‍💻 Author

**Priya Singh**  
[GitHub](https://github.com/Priya-singh03) • [LinkedIn](https://linkedin.com/in/priya-singh) • [LeetCode](https://leetcode.com/u/Priya_singh_23)
