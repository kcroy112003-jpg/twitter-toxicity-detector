from utils import predict_toxicity, load_data
import joblib
import sys

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load dummy data
df = load_data('dummy_tweets.csv')

print("Twitter Toxicity Classification CLI")
print(f"Dataset: {len(df)} tweets")
toxic_count = df['oh_label'].sum()
non_toxic_count = len(df) - toxic_count
print(f"Toxic: {toxic_count}, Non-Toxic: {non_toxic_count}")

if len(sys.argv) > 1:
    tweet = ' '.join(sys.argv[1:])
else:
    print("Usage: python cli.py 'your tweet here'")
    sys.exit(1)

if not tweet.strip():
    print("Please enter a valid tweet.")
    sys.exit(1)
if len(tweet) > 280:
    print("Tweet too long (max 280 chars).")
    sys.exit(1)

try:
    label, confidence = predict_toxicity(tweet, model, vectorizer)
    print(f"Tweet: {tweet}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}")
    if label == "Toxic":
        print("⚠️ This tweet is classified as toxic.")
    else:
        print("✅ This tweet is classified as non-toxic.")
except Exception as e:
    print(f"Error: {e}")