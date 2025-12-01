from utils import predict_toxicity
import joblib

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Test with a sample tweet
test_tweet = "Women are terrible drivers"
label, confidence = predict_toxicity(test_tweet, model, vectorizer)
print(f"Tweet: {test_tweet}")
print(f"Prediction: {label}, Confidence: {confidence:.2f}")

# Another test
test_tweet2 = "Have a great day!"
label2, confidence2 = predict_toxicity(test_tweet2, model, vectorizer)
print(f"Tweet: {test_tweet2}")
print(f"Prediction: {label2}, Confidence: {confidence2:.2f}")