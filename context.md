# Twitter Toxicity Classification Project - Context for LLM Report Generation

## Project Overview
This project implements a machine learning-based system for classifying Twitter tweets as toxic or non-toxic using Natural Language Processing (NLP) techniques. The core functionality is wrapped in a Streamlit web application for user-friendly interaction. The project uses a dummy dataset for demonstration purposes, maintaining realistic preprocessing and modeling pipelines.

## Dataset Description
- **File**: `dummy_tweets.csv`
- **Size**: 30 rows (21 toxic, 3 neutral, 6 positive)
- **Columns**:
  - `index`: Sequential integer (1-30)
  - `id`: Same as index
  - `Text`: Raw tweet content (strings, up to 280 characters)
  - `Annotation`: Categorical label ('sexism', 'racism', 'none', 'positive')
  - `oh_label`: Binary label (1 for toxic, 0 for non-toxic)
- **Distribution**: 70% toxic (oh_label=1), 10% neutral (oh_label=0, Annotation='none'), 20% positive (oh_label=0, Annotation='positive')
- **Content Examples**:
  - Toxic: "Women are terrible drivers, that's why they shouldn't be allowed on roads." (sexism, 1)
  - Neutral: "Just eating breakfast this morning." (none, 0)
  - Positive: "Spread love and kindness everywhere!" (positive, 0)
- **Purpose**: Showcase dataset for training and visualization; imbalanced to test model robustness.

## Data Preprocessing Pipeline
Implemented in `utils.py` via `preprocess_text()` function:
1. **Lowercasing**: Convert all text to lowercase.
2. **URL Removal**: Regex `r'http\S+'` to remove links.
3. **Mention Removal**: Regex `r'@\w+'` to remove @usernames.
4. **Punctuation Removal**: Use `string.punctuation` to strip symbols.
5. **Tokenization**: NLTK `word_tokenize()` (requires punkt/punkt_tab).
6. **Stopword Removal**: NLTK stopwords (English) to filter common words.
7. **Lemmatization**: NLTK WordNetLemmatizer to reduce words to base forms.
8. **Output**: Space-separated string of cleaned tokens.

## Feature Extraction
- **Method**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Implementation**: `sklearn.feature_extraction.text.TfidfVectorizer`
- **Parameters**: `max_features=5000` to limit vocabulary.
- **Process**: Fit on preprocessed texts, transform to sparse matrix for model input.

## Model Training
- **Algorithm**: Logistic Regression (binary classification)
- **Library**: `sklearn.linear_model.LogisticRegression`
- **Parameters**: `class_weight='balanced'` to handle imbalance.
- **Training Process** (`train.py`):
  1. Load dummy data.
  2. Preprocess all texts.
  3. Extract TF-IDF features.
  4. Split data: 80% train, 20% test (`train_test_split`).
  5. Fit model on training set.
  6. Predict on test set.
  7. Evaluate: Accuracy, precision, recall, F1-score, confusion matrix.
- **Saved Artifacts**:
  - `model.pkl`: Trained LogisticRegression model.
  - `vectorizer.pkl`: Fitted TfidfVectorizer.
- **Performance (on dummy data)**: Accuracy ~67%, with warnings for small test set (6 samples).

## Model Evaluation
- **Metrics** (from `sklearn.metrics`):
  - Accuracy: Overall correct predictions.
  - Precision/Recall/F1: Per-class scores.
  - Confusion Matrix: True positives, false positives, etc.
- **Output Example** (from training):
  ```
  Accuracy: 0.6666666666666666
                precision    recall  f1-score   support
           0       0.00      0.00      0.00         2
           1       0.67      1.00      0.80         4
  ```
- **Limitations**: Small dataset leads to poor minority class performance; real-world use requires larger, balanced data.

## Prediction Pipeline
- **Function**: `predict_toxicity(text, model, vectorizer)`
- **Steps**:
  1. Preprocess input text.
  2. Transform to TF-IDF vector.
  3. Predict class (0/1) and probabilities.
  4. Return label ("Toxic"/"Non-Toxic") and confidence (max probability).
- **Edge Cases Handled**:
  - Empty input: Warning.
  - Long text (>280 chars): Truncation warning.
  - Model/vectorizer load failure: Error message.

## Web Application (Streamlit)
- **File**: `app.py`
- **Framework**: Streamlit 1.51.0
- **Features**:
  - **Title**: "🕵️ Twitter Toxicity Classification"
  - **Sidebar**: Dataset stats (pie chart for toxicity distribution).
  - **Main Area** (Two Columns):
    - Left: Text input, predict button, results (color-coded), input word frequency bar chart.
    - Right: Tabs for visuals.
  - **Tabs**:
    - **Word Clouds**: Toxic and non-toxic word clouds from dataset.
    - **Sentiment**: Histogram of TextBlob polarity scores.
    - **Model Metrics**: Placeholder for confusion matrix.
- **Visuals**:
  - Pie Chart: Toxicity distribution (matplotlib/seaborn).
  - Word Clouds: Generated with `wordcloud` library.
  - Bar Charts: Word frequencies (matplotlib).
  - Histograms: Sentiment polarity.
- **Interactivity**: Real-time prediction, dynamic input analysis.
- **Error Handling**: Try-except for loading and prediction.

## Additional Utilities
- **Word Cloud Generation**: `generate_wordcloud(texts, title)` - Creates matplotlib figure with WordCloud.
- **Sentiment Analysis**: `analyze_sentiment(texts)` - Returns polarity list using TextBlob.
- **Keyword Extraction**: `extract_keywords(text)` - Returns Counter of word frequencies.
- **Confusion Matrix**: `get_confusion_matrix(y_true, y_pred)` - Seaborn heatmap.

## Setup and Environment
- **Python Version**: 3.11.10 (via Miniconda)
- **Virtual Environment**: `streamlit_env311` (created with `python -m venv`)
- **Dependencies** (from `requirements.txt`):
  - streamlit
  - pandas
  - nltk
  - scikit-learn
  - matplotlib
  - seaborn
  - textblob
  - joblib
  - wordcloud
- **NLTK Data**: stopwords, punkt, punkt_tab, wordnet (downloaded automatically)
- **Installation Commands**:
  ```
  python -m venv streamlit_env311
  streamlit_env311\Scripts\activate
  pip install -r requirements.txt
  python -c "import nltk; [nltk.download(pkg) for pkg in ['stopwords', 'punkt', 'punkt_tab', 'wordnet']]"
  python train.py
  streamlit run app.py
  ```
- **Auto-Launch**: `setup.bat` activates venv and runs app.

## File Structure
```
D:\Twitter-Toxicity-Classification-main\
├── app.py                 # Streamlit web app
├── cli.py                 # Command-line interface
├── dummy_tweets.csv       # Dummy dataset
├── model.pkl              # Trained model
├── nltk_download.py       # NLTK data downloader
├── requirements.txt       # Dependencies
├── setup.bat              # Auto-launch script
├── train.py               # Training script
├── utils.py               # Utility functions
└── README.md              # Documentation
```

## Key Insights and Analysis
- **ML Pipeline Robustness**: Handles text noise (URLs, mentions) well; TF-IDF captures semantic differences.
- **Imbalance Handling**: Class weights improve recall for minority class.
- **Web App Usability**: Intuitive UI with visuals aids understanding; real-time feedback.
- **Performance**: Suitable for demo; production needs larger dataset and hyperparameter tuning.
- **Ethical Considerations**: Binary toxicity classification; annotations include sexism/racism; positive labels for balance.
- **Extensibility**: Easy to add models (e.g., BERT), features (e.g., emoji analysis), or deploy (e.g., Heroku).

## Potential Report Sections for LLM
1. **Executive Summary**: Project goals, outcomes.
2. **Data Analysis**: Dataset stats, preprocessing impact.
3. **Model Development**: Training process, evaluation metrics.
4. **Web Application Review**: UI/UX, features, visuals.
5. **Technical Implementation**: Code structure, dependencies.
6. **Performance Evaluation**: Strengths, weaknesses, improvements.
7. **Deployment and Usage**: Setup, running the app.
8. **Future Enhancements**: Scalability, advanced models.
9. **Conclusion**: Overall assessment.

This context provides comprehensive details for generating a detailed report on the codebase, covering ML training, analysis, and web app aspects.</content>
<parameter name="filePath">D:\Twitter-Toxicity-Classification-main\context.md