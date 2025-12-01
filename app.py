import streamlit as st
import joblib
from utils import predict_toxicity, load_data, generate_wordcloud, analyze_sentiment, extract_keywords, get_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load model and vectorizer with error handling
try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please train the model first.")
    st.stop()

# Load dummy data for visualizations
try:
    df = load_data('dummy_tweets.csv')
except FileNotFoundError:
    st.error("Dummy data file not found.")
    st.stop()

st.title("🕵️ Twitter Toxicity Classification")

st.sidebar.header("📊 Dataset Overview")
st.sidebar.write(f"Total tweets: {len(df)}")
toxic_count = df['oh_label'].sum()
non_toxic_count = len(df) - toxic_count
st.sidebar.write(f"🔴 Toxic: {toxic_count}, 🟢 Non-Toxic: {non_toxic_count}")

# Pie chart for distribution
fig_pie, ax_pie = plt.subplots()
ax_pie.pie([non_toxic_count, toxic_count], labels=['Non-Toxic', 'Toxic'], autopct='%1.1f%%', colors=['green', 'red'])
ax_pie.set_title('Toxicity Distribution')
st.sidebar.pyplot(fig_pie)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Classify Your Tweet")
    tweet = st.text_area("Enter a tweet to classify:", height=100)

    if st.button("🚀 Predict Toxicity"):
        if not tweet.strip():
            st.warning("⚠️ Please enter a tweet.")
        elif len(tweet) > 280:
            st.warning("⚠️ Tweet too long (max 280 chars).")
        else:
            try:
                label, confidence = predict_toxicity(tweet, model, vectorizer)
                if label == "Toxic":
                    st.error(f"🔴 Prediction: {label} (Confidence: {confidence:.2f})")
                else:
                    st.success(f"🟢 Prediction: {label} (Confidence: {confidence:.2f})")

                # Input analysis
                st.subheader("📝 Input Analysis")
                keywords = extract_keywords(tweet)
                if keywords:
                    top_words = dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10])
                    fig_bar, ax_bar = plt.subplots()
                    ax_bar.bar(top_words.keys(), top_words.values(), color='skyblue')
                    ax_bar.set_title('Top Words in Your Tweet')
                    ax_bar.set_ylabel('Frequency')
                    plt.xticks(rotation=45)
                    st.pyplot(fig_bar)
                else:
                    st.write("No significant words detected.")

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

with col2:
    st.header("📈 Visual Insights")

    tab1, tab2, tab3 = st.tabs(["Word Clouds", "Sentiment", "Model Metrics"])

    with tab1:
        st.subheader("Word Clouds from Dataset")
        toxic_texts = df[df['oh_label'] == 1]['Text'].tolist()
        non_toxic_texts = df[df['oh_label'] == 0]['Text'].tolist()

        if toxic_texts:
            st.pyplot(generate_wordcloud(toxic_texts, "Toxic Words"))
        if non_toxic_texts:
            st.pyplot(generate_wordcloud(non_toxic_texts, "Non-Toxic Words"))

    with tab2:
        st.subheader("Sentiment Analysis")
        polarities = analyze_sentiment(df['Text'].tolist())
        fig_sent, ax_sent = plt.subplots()
        ax_sent.hist(polarities, bins=20, color='purple', alpha=0.7)
        ax_sent.set_title('Sentiment Polarity Distribution')
        ax_sent.set_xlabel('Polarity')
        ax_sent.set_ylabel('Frequency')
        st.pyplot(fig_sent)

    with tab3:
        st.subheader("Confusion Matrix (Training)")
        # Since we don't have y_true/y_pred saved, simulate or note
        st.write("Confusion matrix from training (placeholder - would show actual metrics)")
        # If we had saved, load; for now, placeholder
        fig_cm = plt.figure(figsize=(6,4))
        plt.text(0.5, 0.5, 'Confusion Matrix\n(Actual metrics from training)', ha='center', va='center', fontsize=12)
        plt.axis('off')
        st.pyplot(fig_cm)