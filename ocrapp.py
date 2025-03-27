import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
import re

# Set Tesseract Path (Update this for deployment)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Adjust path if needed

st.set_page_config(page_title="OCR & Structured Data Extraction", layout="wide")

# Function to extract text using Tesseract OCR
def extract_text(image):
    text = pytesseract.image_to_string(image)
    return text.strip()

# Function to extract structured information
def extract_structured_data(text):
    structured_data = {
        "Patient Info": {},
        "Vitals": {},
        "Diagnosis": [],
        "Medications": [],
        "Investigations": [],
        "Advice": [],
        "Follow-Up": {}
    }

    # Extract Patient Info
    age_match = re.search(r'(\d{1,2})\s*years?\s*old', text, re.IGNORECASE)
    structured_data["Patient Info"]["Age"] = age_match.group(1) if age_match else "Not found"

    gender_match = re.search(r'\b(male|female|M|F|other)\b', text, re.IGNORECASE)
    structured_data["Patient Info"]["Gender"] = gender_match.group(1) if gender_match else "Not found"

    weight_match = re.search(r'(\d{2,3})\s*kg', text, re.IGNORECASE)
    structured_data["Patient Info"]["Weight"] = weight_match.group(1) + " kg" if weight_match else "Not found"

    # Extract Vitals
    bp_match = re.search(r'(\d{2,3}/\d{2,3})', text)
    structured_data["Vitals"]["Blood Pressure"] = bp_match.group(1) if bp_match else "Not found"

    # Extract Diagnosis
    diagnosis_match = re.findall(r"(?<=Diagnosis:)(.*)", text, re.IGNORECASE)
    structured_data["Diagnosis"] = [d.strip() for d in diagnosis_match if d.strip()]

    # Extract Medications
    med_match = re.findall(r'\b(?:TAB|Tablet|CAP|Capsule|Syrup|Injection)\s+([A-Za-z0-9\s\(\)-]+)', text)
    structured_data["Medications"] = [m.strip() for m in med_match if m.strip()]

    # Extract Advice
    advice_match = re.findall(r'(\*.*)', text)
    structured_data["Advice"] = [a.strip() for a in advice_match if a.strip()]

    # Extract Follow-Up Date
    followup_match = re.search(r'Follow[- ]?Up:\s*(\d{2}-\d{2}-\d{4})', text)
    structured_data["Follow-Up"]["Date"] = followup_match.group(1) if followup_match else "Not found"

    return structured_data

# Function to display structured data beautifully
def display_structured_output(structured_data):
    st.markdown("## 📑 Structured Medical Data")

    # Patient Info
    with st.expander("👤 Patient Information", expanded=True):
        st.table(pd.DataFrame([structured_data["Patient Info"]]))

    # Vitals
    with st.expander("🩺 Vitals", expanded=True):
        st.table(pd.DataFrame([structured_data["Vitals"]]))

    # Diagnosis
    with st.expander("🦠 Diagnosis", expanded=True):
        if structured_data["Diagnosis"]:
            st.write("\n".join(f"- {d}" for d in structured_data["Diagnosis"]))
        else:
            st.write("No diagnosis found.")

    # Medications
    with st.expander("💊 Medications", expanded=True):
        if structured_data["Medications"]:
            st.write("\n".join(f"- {m}" for m in structured_data["Medications"]))
        else:
            st.write("No medications found.")

    # Advice
    with st.expander("📌 Advice", expanded=True):
        if structured_data["Advice"]:
            st.write("\n".join(f"- {a}" for a in structured_data["Advice"]))
        else:
            st.write("No advice found.")

    # Follow-Up
    with st.expander("📅 Follow-Up", expanded=True):
        st.write(f"**Date:** {structured_data['Follow-Up'].get('Date', 'Not found')}")

# Sidebar for extracted details
def display_sidebar(structured_data):
    with st.sidebar:
        st.subheader("🔍 Quick View")
        st.write(f"**Age:** {structured_data['Patient Info']['Age']}")
        st.write(f"**Gender:** {structured_data['Patient Info']['Gender']}")
        st.write(f"**Weight:** {structured_data['Patient Info']['Weight']}")
        st.write(f"**Blood Pressure:** {structured_data['Vitals']['Blood Pressure']}")
        st.write(f"**Follow-Up Date:** {structured_data['Follow-Up'].get('Date', 'Not found')}")

# Streamlit App
def main():
    st.title("🩺 OCR-Based Medical Document Processor")
    st.write("Upload an image of a **prescription or medical document**, and I will extract the text and structure the data!")

    uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        if st.button("🚀 SUBMIT"):
            st.markdown("### 📝 Extracted Text")
            extracted_text = extract_text(image)
            st.text_area("Raw OCR Output", extracted_text, height=200)

            # Process structured data
            structured_data = extract_structured_data(extracted_text)

            # Display structured output beautifully
            display_structured_output(structured_data)

            # Sidebar for quick summary
            display_sidebar(structured_data)

if __name__ == "__main__":
    main()
