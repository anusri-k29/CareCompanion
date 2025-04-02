import streamlit as st
import cv2
import pytesseract
import numpy as np
import json
import re
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract" 

import streamlit as st
import cv2
import pytesseract
import numpy as np
import re
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Enhanced Medical Regex Patterns
MEDICAL_PATTERNS = {
    'patient': {
        'age_gender_pattern1': r'PATIENT\s*\(\s*(?P<gender1>M|F|Male|Female)\s*\)\s*/\s*(?P<age1>\d{1,3})(?=Y\b)',
        'age_gender_pattern2': r',\s*(?P<age2>\d{1,3})\s*/\s*(?P<gender2>M|F|Male|Female)\b'
    },
    'patient_extra': {
        'weight': r'Weight\s*\(Kg\)\s*:\s*(\d+)',
        'health_card': r'Health\s*Card[:\s]*Exp[:\s]*(\d{4}[\/\-]\d{2}[\/\-]\d{2})'
    },
    'clinical': {
        'diagnosis': r'(?i)Diagnosis[:\s-]+([\s\S]+?)(?=\n\s*\n|Medicine Name)',
        'vitals': {
            'bp': r'(?i)(?:BP|Blood\s*Pressure)[\s:]*(\d{2,3}\s*/\s*\d{2,3})\s*(?:mmHg)?',
            'pulse': r'(?i)(?:Pulse|Heart\s*Rate)[\s:]*(\d{2,3})\s*(?:bpm)?',
            'temp': r'(?i)(?:Temp|Temperature)[\s:]*(\d{2}\.?\d*)\s*°?[CF]?',
            'rr': r'(?i)(?:RR|Respiratory\s*Rate)[\s:]*(\d{2})\s*(?:/min)?',
            'spo2': r'(?i)(?:SpO2|Oxygen\s*Saturation)[\s:]*(\d{2,3})\s*%?'
        },
        'complaints': r'(?i)Chief\s*Complaints[:\s-]+([\s\S]+?)(?=\n)',
        'reactions': r'(?i)(?:Adverse\s*Reactions)[\s:]+([\s\S]+?)(?=\n)',
        'investigations': r'(?i)(?:Investigations|Tests)[:\s-]+([\s\S]+?)(?=\n\s*\n|Medicine|Advice|$)'
    },
    'medications': {
        'pattern': r'(?m)^\s*\d+\)\s*((?:(?!^\s*\d+\)).)+)'
    },
    'advice': r'(?i)Advice[:\s-]+([\s\S]+?)(?=\n\s*(?:Follow\s*Up|Next\s*Visit)|$)',
    'follow_up': r'(?i)Follow\s*Up[:\s-]+(\d{2}[\/\-]\d{2}[\/\-]\d{2,4})'
} 

class OCRProcessor:
    def extract_text(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray)

class MedicalDataExtractor:
    def extract_medical_data(self, text):
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        result = {"patient": {}, "vitals": {}, "diagnosis": [], "medications": [], "advice": [], "follow_up": {}}
        
        # Extract patient details
        age, gender = self.extract_age_gender(text)
        if age and gender:
            result["patient"].update({"age": age, "gender": gender})
        
        # Extract vitals and other medical data
        result["vitals"] = self.extract_vitals(text)
        result["diagnosis"] = self.extract_list_data(text, MEDICAL_PATTERNS['clinical']['diagnosis'])
        result["medications"] = self.extract_list_data(text, MEDICAL_PATTERNS['medications']['pattern'])
        result["advice"] = self.extract_list_data(text, MEDICAL_PATTERNS['advice'])
        result["follow_up"]["date"] = self.extract_follow_up(text)
        
        return result

    def extract_list_data(self, text, pattern):
        match = re.search(pattern, text)
        return [d.strip() for d in match.group(1).split('\n') if d.strip()] if match else []

    def extract_follow_up(self, text):
        match = re.search(MEDICAL_PATTERNS['follow_up'], text)
        return match.group(1).strip() if match else None

# Streamlit UI
st.set_page_config(page_title="MedScan Pro", layout="wide")
st.title("🩺 MedScan Pro - Smart Medical Document Analysis")
st.markdown("Upload medical images or PDFs to extract structured healthcare data")

uploaded_files = st.file_uploader("Upload medical documents", type=["jpg", "png", "jpeg", "pdf"], accept_multiple_files=True)

if uploaded_files:
    ocr_processor = OCRProcessor()
    data_extractor = MedicalDataExtractor()
    extracted_texts = []
    
    for file in uploaded_files:
        image = Image.open(file)
        img_array = np.array(image)
        extracted_texts.append(ocr_processor.extract_text(img_array))
    
    full_text = "\n".join(extracted_texts)
    structured_data = data_extractor.extract_medical_data(full_text)
    
    st.success("Analysis Complete!")
    st.balloons()
    
    st.subheader("Structured Data Output")
    st.json(structured_data)
    
    if st.checkbox("Show Raw OCR Text"):
        st.subheader("Raw OCR Output")
        st.code(full_text)

st.markdown("---")
st.markdown("*Medical data extraction powered by Tesseract OCR and advanced NLP patterns*")

