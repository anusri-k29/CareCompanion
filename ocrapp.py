import streamlit as st
import cv2
import pytesseract
import numpy as np
import json
import re
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Adjust path as needed

# Function to perform OCR on an image
def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

# Function to extract structured information from text
def extract_structured_data(text):
    structured_data = {
        "patient": {},
        "vitals": {},
        "diagnosis": [],
        "medications": [],
        "investigations": [],
        "advice": [],
        "follow_up": {}
    }
    
    # Extract Age
    age_match = re.search(r'Age[:\s]*([0-9]+)', text, re.IGNORECASE)
    if age_match:
        structured_data["patient"]["age"] = age_match.group(1)
    
    # Extract Gender
    gender_match = re.search(r'\b(Male|Female|M|F)\b', text, re.IGNORECASE)
    if gender_match:
        structured_data["patient"]["gender"] = gender_match.group(1)
    
    # Extract Blood Pressure
    bp_match = re.search(r'(\d{2,3}/\d{2,3})', text)
    if bp_match:
        structured_data["vitals"]["bp"] = bp_match.group(1)
    
    # Extract Diagnosis
    diag_match = re.findall(r'Diagnosis[:\n\s]*(.*)', text, re.IGNORECASE)
    structured_data["diagnosis"] = diag_match
    
    # Extract Medications
    med_match = re.findall(r'((?:TAB|Tablet|CAP|Capsule|Syrup|Injection).*?(?:\d+\s*(?:mg|g|ml|Morning|Night|Days|After Food|Tot:\d+)))', text, re.IGNORECASE)
    structured_data["medications"] = [m.strip() for m in med_match if m.strip()]
    
    # Extract Advice
    advice_match = re.findall(r'\*\s*(.+)', text)
    structured_data["advice"] = advice_match
    
    # Extract Follow-up Date
    follow_up_match = re.search(r'Follow Up[:\s]*(\d{2}-\d{2}-\d{4})', text)
    if follow_up_match:
        structured_data["follow_up"]["date"] = follow_up_match.group(1)
    
    return structured_data

# Streamlit UI
st.title("OCR-Based Medical Document Analyzer")
st.write("Upload a medical prescription image to extract text and structured data.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("SUBMIT"):
        extracted_text = extract_text(image)
        structured_data = extract_structured_data(extracted_text)
        
        st.subheader("Extracted Text")
        st.text_area("", extracted_text, height=200)
        
        st.subheader("Structured Medical Data")
        st.json(structured_data)
