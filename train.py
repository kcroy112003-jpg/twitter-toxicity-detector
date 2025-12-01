from utils import load_data, preprocess_text, extract_features, train_model

# Load dummy data
df = load_data('dummy_tweets.csv')

# Preprocess texts
df['processed'] = df['Text'].apply(preprocess_text)

# Extract features
X, vectorizer = extract_features(df['processed'])

# Train model
model = train_model(X, df['oh_label'])

# Save model and vectorizer
import joblib
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved.")