import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# App title
st.title("📰 Fake News Detection App")

# Text input
user_input = st.text_area("Enter the news article content here 👇", height=250)

# Predict button
if st.button("Check if it's Fake or Real"):
    if user_input.strip() == "":
        st.warning("Please enter some text to check.")
    else:
        # Preprocess and vectorize user input
        transformed_input = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(transformed_input)
        proba = model.predict_proba(transformed_input)

        # Display result
        if prediction[0] == 1:
            st.success("✅ This news is predicted as **Real**.")
        else:
            st.error("🚨 This news is predicted as **Fake**.")

        st.write(f"**Confidence:** {proba[0][prediction[0]]:.2f}")

# Footer
st.markdown("---")
st.write("Made with ❤️ by Iman Adhikary")
