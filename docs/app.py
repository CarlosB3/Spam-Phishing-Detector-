import streamlit as st
import joblib
import os

st.write(os.getcwd())
st.write(os.listdir())

# Load the trained model and TF-IDF vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Phish Detector",
    page_icon="📧",
    layout="centered"
)

# Title
st.title("📧 Phish Detector")
st.subheader("Spam & Phishing Message Classifier")

st.write(
    "Enter an email or text message below to determine whether it is "
    "predicted to be **Spam** or **Ham (Legitimate)**."
)

# User input
message = st.text_area(
    "Enter your message:",
    height=200,
    placeholder="Type or paste a message here..."
)

# Analyze button
if st.button("Analyze Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        # Transform the message
        message_vector = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(message_vector)[0]

        # Prediction probabilities
        probabilities = model.predict_proba(message_vector)[0]

        ham_confidence = probabilities[0] * 100
        spam_confidence = probabilities[1] * 100

        st.divider()

        if prediction == 1:
            st.error("🚨 Prediction: SPAM")
            st.write(f"**Confidence:** {spam_confidence:.2f}%")
        else:
            st.success("✅ Prediction: HAM (Legitimate)")
            st.write(f"**Confidence:** {ham_confidence:.2f}%")

        st.divider()

        st.subheader("Prediction Confidence")

        st.write(f"Spam: **{spam_confidence:.2f}%**")
        st.progress(float(spam_confidence) / 100)

        st.write(f"Ham: **{ham_confidence:.2f}%**")
        st.progress(float(ham_confidence) / 100)
