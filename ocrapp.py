import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
import re

# Set Tesseract Path (Update this for deployment if necessary)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Adjust path as needed

st.set_page_config(page_title="OCR & Structured Data Extraction", layout="wide")

# Function to extract text using Tesseract OCR
def extract_text(image):
    text = pytesseract.image_to_string(image)
    return text.strip()

# Function to extract structured information using regex
def extract_structured_data(text):
    structured_data = {}

    # Extract Age
    age_match = re.search(r'(\d{1,2})\s*years?\s*old', text, re.IGNORECASE)
    structured_data["Age"] = age_match.group(1) if age_match else "Not found"

    # Extract Gender
    if re.search(r'\b(male|female|other)\b', text, re.IGNORECASE):
        structured_data["Gender"] = re.search(r'\b(male|female|other)\b', text, re.IGNORECASE).group(1)
    else:
        structured_data["Gender"] = "Not found"

    # Extract Medications
    meds_match = re.findall(r'\b(?:Tab|Tablet|Cap|Capsule|Syrup|Injection)\s+([A-Za-z0-9]+)', text)
    structured_data["Medications"] = ", ".join(meds_match) if meds_match else "Not found"

    return structured_data

# Function to display structured data beautifully
def display_structured_output(structured_data):
    st.markdown("## 📑 Extracted Structured Information")
    
    if structured_data:
        df = pd.DataFrame(structured_data.items(), columns=["Field", "Value"])
        st.table(df)  # Nicely formatted table
    
    st.markdown("### 🗂️ JSON View")
    st.json(structured_data, expanded=False)  # Collapsible JSON display

# Sidebar for extra details
def display_sidebar(structured_data):
    with st.sidebar:
        st.subheader("🔍 Extracted Details")
        for key, value in structured_data.items():
            st.write(f"**{key}:** {value}")

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

            # Sidebar for extra details
            display_sidebar(structured_data)

if __name__ == "__main__":
    main()
