# Twitter Toxicity Classification - Streamlit Web App

This project implements a Streamlit web app for identifying the toxicity of tweets using Natural Language Processing (NLP) techniques. It classifies tweets into binary values (toxic or non-toxic) based on their content.

## Problem Statement

The app allows users to input tweets and receive real-time toxicity predictions, helping to flag harmful content on social media.

## Dataset

For demonstration, we use a dummy dataset `dummy_tweets.csv` with 30 sample tweets (70% toxic, 10% neutral, 20% positive). It consists of the following columns:

- `index`: Index of the tweet
- `id`: Unique identifier for each tweet
- `Text`: The text content of the tweet
- `Annotation`: Annotation indicating the toxicity level (e.g., 'sexism', 'none', 'positive')
- `oh_label`: Binary label indicating the toxicity (1 for toxic, 0 for non-toxic)

## Approach

The app uses an NLP pipeline for toxicity classification:

1. **Data Preparation**: Dummy dataset with balanced toxic/non-toxic samples.

2. **Text Preprocessing**: Lowercasing, removing URLs/mentions/punctuation, tokenization, stopword removal, lemmatization.

3. **Feature Extraction**: TF-IDF vectorization for numerical features.

4. **Model Training**: Logistic Regression with class weights for imbalance.

5. **Web App**: Streamlit interface for input, prediction, and visualizations.

## Installation and Setup

To run the Streamlit web app locally, follow these steps:

1. Clone this repository:
   ```
   git clone https://github.com/tushar2704/twitter-toxicity-classification.git
   cd twitter-toxicity-classification
   ```

2. Set up a virtual environment with Python 3.11:
   ```
   python -m venv streamlit_env
   streamlit_env\Scripts\activate  # On Windows
   ```

3. Install the required dependencies:
   ```
   pip install streamlit pandas nltk scikit-learn matplotlib seaborn textblob joblib
   ```

4. Download NLTK data:
   ```
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"
   ```

5. Train the model (if not already done):
   ```
   python train.py
   ```

6. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

The app will open in your browser. Enter a tweet to get toxicity predictions.

## Features

- Real-time tweet toxicity prediction.
- Confidence scores and visual feedback.
- Dataset statistics and bar chart visualization.
- Error handling for invalid inputs.

## Contributing

Contributions are welcome. Open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Inspired by NLP techniques for content moderation.

## Contact

For inquiries, contact [Tushar Aggarwal](mailto:info@tushar-aggarwal.com).