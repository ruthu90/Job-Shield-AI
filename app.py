import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="JOB SHIELD AI", layout="wide")

st.title("🔐 JOB SHIELD AI")
st.markdown("### Intelligent Fraud Detection for Websites & Job Descriptions")

# ---------------------------------------------------
# TRAIN SIMPLE NLP MODEL (runs instantly)
# ---------------------------------------------------

training_texts = [
    "win money fast",
    "urgent bank verification",
    "claim your lottery prize",
    "send account details immediately",
    "guaranteed investment return",
    "official company website",
    "about our services",
    "contact customer support",
    "job application form",
    "career opportunity hiring now"
]

labels = [1,1,1,1,1,0,0,0,0,0]  # 1 = Fraud, 0 = Genuine

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_texts)

model = LogisticRegression()
model.fit(X, labels)

# ---------------------------------------------------
# FUNCTION TO ANALYZE TEXT
# ---------------------------------------------------

def analyze_text(text):
    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)[0]
    probability = model.predict_proba(transformed)[0][1]

    return prediction, probability

# ---------------------------------------------------
# INPUT SELECTION
# ---------------------------------------------------

option = st.radio(
    "Select Input Type",
    ("Website URL", "Data Link", "Job Description Text")
)

input_text = ""

# ---------------------------------------------------
# WEBSITE URL CHECK
# ---------------------------------------------------

if option == "Website URL":
    url = st.text_input("Enter Website URL (include https://)")
    
    if st.button("Analyze Website"):
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            input_text = soup.get_text()[:1500]

            prediction, probability = analyze_text(input_text)

            if prediction == 1:
                st.error(f"⚠ Fraud Risk Detected ({round(probability*100,2)}% confidence)")
                st.write("Suspicious keywords or financial traps detected.")
            else:
                st.success(f"✅ Website Appears Genuine ({round((1-probability)*100,2)}% confidence)")
                st.write("Content appears informational and professional.")

        except:
            st.error("Unable to fetch website. Check the URL.")

# ---------------------------------------------------
# DATA LINK CHECK
# ---------------------------------------------------

elif option == "Data Link":
    data_link = st.text_input("Enter Data/Drive/Dropbox Link")

    if st.button("Analyze Link"):
        prediction, probability = analyze_text(data_link)

        if prediction == 1:
            st.error(f"⚠ Suspicious Link Detected ({round(probability*100,2)}% confidence)")
            st.write("Link contains patterns commonly used in phishing or scams.")
        else:
            st.success(f"✅ Link Looks Safe ({round((1-probability)*100,2)}% confidence)")
            st.write("No suspicious patterns detected in link structure.")

# ---------------------------------------------------
# JOB DESCRIPTION CHECK
# ---------------------------------------------------

elif option == "Job Description Text":
    job_text = st.text_area("Paste Job Description Here")

    if st.button("Analyze Job Description"):
        prediction, probability = analyze_text(job_text)

        if prediction == 1:
            st.error(f"⚠ Potential Fraud Job ({round(probability*100,2)}% confidence)")
            st.write("This job description contains urgent money requests or suspicious language.")
        else:
            st.success(f"✅ Job Appears Genuine ({round((1-probability)*100,2)}% confidence)")
            st.write("The job description looks legitimate based on content analysis.")

# ---------------------------------------------------
# FEEDBACK SECTION
# ---------------------------------------------------

st.markdown("---")
st.subheader("📝 User Feedback")

name = st.text_input("Your Name")
feedback = st.text_area("Share your feedback about JOB SHIELD AI")

if st.button("Submit Feedback"):
    if name and feedback:
        new_data = pd.DataFrame([[name, feedback]], columns=["Name", "Feedback"])

        try:
            old_data = pd.read_csv("feedback.csv")
            updated = pd.concat([old_data, new_data], ignore_index=True)
        except:
            updated = new_data

        updated.to_csv("feedback.csv", index=False)
        st.success("Thank you for your feedback!")

# ---------------------------------------------------
# DISPLAY FEEDBACK TABLE
# ---------------------------------------------------

st.subheader("📊 Community Feedback")

try:
    feedback_data = pd.read_csv("feedback.csv")
    st.dataframe(feedback_data)
except:
    st.write("No feedback submitted yet.")