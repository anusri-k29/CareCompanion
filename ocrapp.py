import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import json
import glob
import os

# --- Enhanced Medical Regex Patterns ---
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

# --- OCR and Extraction Classes ---

class OCRProcessor:
    """Extracts text from an image using Tesseract OCR."""
    
    def extract_text(self, image: Image.Image) -> str:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        # Convert color space for OpenCV (RGB to BGR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text


class MedicalDataExtractor:
    """Extracts structured medical data from OCR text."""
    
    def extract_age_gender(self, text: str):
        m1 = re.search(MEDICAL_PATTERNS['patient']['age_gender_pattern1'], text)
        if m1:
            return m1.group('age1').strip(), m1.group('gender1').strip()
        m2 = re.search(MEDICAL_PATTERNS['patient']['age_gender_pattern2'], text)
        if m2:
            return m2.group('age2').strip(), m2.group('gender2').strip()
        return None, None

    def extract_vitals(self, text: str):
        vitals = {}
        for vital, pattern in MEDICAL_PATTERNS['clinical']['vitals'].items():
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip()
                if vital == 'bp':
                    value = re.sub(r'\s+', '', value)
                elif vital in ['temp', 'spo2']:
                    value = value.replace(' ', '')
                vitals[vital] = value
        return vitals

    def extract_medical_data(self, text: str):
        # Normalize whitespace (preserving newlines)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        result = {
            "patient": {},
            "vitals": {},
            "diagnosis": [],
            "medications": [],
            "investigations": [],
            "advice": [],
            "follow_up": {}
        }

        try:
            age, gender = self.extract_age_gender(text)
            if age and gender:
                result["patient"]["age"] = age
                result["patient"]["gender"] = gender

            weight = re.search(MEDICAL_PATTERNS['patient_extra']['weight'], text, re.I)
            if weight:
                result["patient"]["weight"] = f"{weight.group(1).strip()} kg"

            result["vitals"] = self.extract_vitals(text)

            diagnosis = re.search(MEDICAL_PATTERNS['clinical']['diagnosis'], text)
            if diagnosis:
                result["diagnosis"] = [line.strip() for line in diagnosis.group(1).split('\n') if line.strip()]

            inv = re.search(MEDICAL_PATTERNS['clinical']['investigations'], text)
            if inv:
                result["investigations"] = [line.strip() for line in inv.group(1).split('\n') if line.strip()]

            meds = re.findall(MEDICAL_PATTERNS['medications']['pattern'], text, re.DOTALL | re.MULTILINE)
            if meds:
                result["medications"].extend([re.sub(r'\s+', ' ', m).strip() for m in meds if m.strip()])

            advice = re.search(MEDICAL_PATTERNS['advice'], text)
            if advice:
                result["advice"] = [line.strip() for line in advice.group(1).split('\n') if line.strip()]

            follow_up = re.search(MEDICAL_PATTERNS['follow_up'], text)
            if follow_up:
                result["follow_up"] = {"date": follow_up.group(1).strip()}

        except Exception as e:
            st.error(f"Extraction error: {str(e)}")
            return None

        return result


class MedicalOCRApp:
    """Streamlit application for processing medical images."""
    
    def __init__(self):
        self.ocr_processor = OCRProcessor()
        self.data_extractor = MedicalDataExtractor()

    def process_image(self, image: Image.Image):
        extracted_text = self.ocr_processor.extract_text(image)
        structured_data = self.data_extractor.extract_medical_data(extracted_text)
        return extracted_text, structured_data

# --- Streamlit UI ---
def main():
    st.title("Medical Prescription Structured Data Extractor")
    st.write("Upload your medical document image(s) and click **SUBMIT** to extract text and structured data.")

    uploaded_files = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    submit = st.button("SUBMIT")

    if submit:
        if not uploaded_files:
            st.warning("Please upload at least one image file.")
        else:
            app = MedicalOCRApp()
            for uploaded_file in uploaded_files:
                st.markdown(f"## Processing Image: **{uploaded_file.name}**")
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_column_width=True)

                # Process the image
                with st.spinner("Processing..."):
                    extracted_text, structured_data = app.process_image(image)

                st.subheader("Extracted OCR Text")
                st.text_area("", extracted_text, height=200)

                st.subheader("Structured Medical Data")
                st.json(structured_data)
                st.markdown("---")

if __name__ == "__main__":
    main()
