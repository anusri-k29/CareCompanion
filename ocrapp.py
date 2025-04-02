import streamlit as st
import cv2
import pytesseract
import numpy as np
import json
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
        'complaints': r'(?i)\b(?:Chief|Primary|Patient)?\s*Complaints?[:\s-]+([\s\S]+?)(?=\n\s*\n|\b(?:Diagnosis|History|Findings|Medications|Investigations|Advice)\b)',
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
        text = pytesseract.image_to_string(gray)
        return text

class MedicalDataExtractor:
    def extract_age_gender(self, text):
        m1 = re.search(MEDICAL_PATTERNS['patient']['age_gender_pattern1'], text)
        if m1: 
            return m1.group('age1'), m1.group('gender1')
        m2 = re.search(MEDICAL_PATTERNS['patient']['age_gender_pattern2'], text)
        if m2: 
            return m2.group('age2'), m2.group('gender2')
        return None, None

    def extract_vitals(self, text):
        vitals = {}
        for vital, pattern in MEDICAL_PATTERNS['clinical']['vitals'].items():
            match = re.search(pattern, text)
            if match: 
                vitals[vital] = match.group(1).strip()
        return vitals

    def extract_medical_data(self, text):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        result = {
            "patient": {}, "vitals": {}, "diagnosis": [],
            "medications": [], "investigations": [],
            "advice": [], "follow_up": {},"complaints": []
        }

        # Extract patient info
        age, gender = self.extract_age_gender(text)
        if age and gender:
            result["patient"]["age"] = age
            result["patient"]["gender"] = gender

        # Extract weight
        weight = re.search(MEDICAL_PATTERNS['patient_extra']['weight'], text, re.I)
        if weight: 
            result["patient"]["weight"] = f"{weight.group(1)} kg"

        # Extract vitals
        result["vitals"] = self.extract_vitals(text)

        # Extract complaints
        complaints = re.search(MEDICAL_PATTERNS['clinical']['complaints'], text)
        if complaints: 
            result["complaints"] = [c.strip() for c in complaints.group(1).split('\n') if c.strip()]

        # Extract diagnosis
        diagnosis = re.search(MEDICAL_PATTERNS['clinical']['diagnosis'], text)
        if diagnosis: 
            result["diagnosis"] = [d.strip() for d in diagnosis.group(1).split('\n') if d.strip()]

        # Extract medications
        meds = re.findall(MEDICAL_PATTERNS['medications']['pattern'], text)
        if meds: 
            result["medications"] = [re.sub(r'\s+', ' ', m).strip() for m in meds]

        # Extract advice
        advice = re.search(MEDICAL_PATTERNS['advice'], text)
        if advice: 
            result["advice"] = [a.strip() for a in advice.group(1).split('\n') if a.strip()]

        # Extract follow-up
        follow_up = re.search(MEDICAL_PATTERNS['follow_up'], text)
        if follow_up: 
            result["follow_up"]["date"] = follow_up.group(1).strip()

        return result

# Streamlit UI
st.set_page_config(page_title="MedScan Pro", layout="wide")
st.title("🩺 MedScan Pro - Smart Medical Document Analysis")
st.markdown("Upload a medical prescription/image to extract structured healthcare data")

with st.sidebar:
    st.header("Settings")
    show_raw_text = st.checkbox("Show raw OCR text", value=False)
    
uploaded_file = st.file_uploader("Upload medical document", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
if uploaded_file:
    ocr_processor = OCRProcessor()
    data_extractor = MedicalDataExtractor()
    
    # Process each image individually
    if st.button("Analyze Documents", type="primary"):
        for file in uploaded_file:
            st.markdown("---")
            st.subheader(f"Results for {file.name}")
            
            image = Image.open(file)
            img_array = np.array(image)
            ocr_text = ocr_processor.extract_text(img_array)
            structured_data = data_extractor.extract_medical_data(ocr_text)
            
            st.image(image, caption=f"Document: {file.name}", use_column_width=True)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("🧑 Patient Information")
                patient_info = structured_data["patient"]
                if patient_info:
                    info_text = f"""
                    - Age: {patient_info.get('age', 'N/A')}
                    - Gender: {patient_info.get('gender', 'N/A')}
                    - Weight: {patient_info.get('weight', 'N/A')}
                    """
                    st.markdown(info_text)
                else:
                    st.warning("No patient information found")
                
                st.subheader("📝 Patient Complaints")
                if structured_data["complaints"]:
                    complaints_md = "\n".join([f"- {c}" for c in structured_data["complaints"]])
                    st.markdown(complaints_md)
                else:
                    st.info("No complaints mentioned")
                
                st.subheader("📈 Vitals")
                if structured_data["vitals"]:
                    vitals_md = "\n".join([f"- {k.upper()}: {v}" for k, v in structured_data["vitals"].items()])
                    st.markdown(vitals_md)
                else:
                    st.info("No vital signs detected")
                    
            with col2:
                tab1, tab2, tab3, tab4 = st.tabs(["Diagnosis", "Medications", "Advice", "Follow Up"])
                with tab1:
                    if structured_data["diagnosis"]:
                        st.markdown("**Primary Diagnosis:**")
                        for dx in structured_data["diagnosis"]:
                            st.markdown(f"- {dx}")
                    else:
                        st.error("No diagnosis information found")
                with tab2:
                    if structured_data["medications"]:
                        for i, med in enumerate(structured_data["medications"], 1):
                            with st.expander(f"Medication #{i}"):
                                st.markdown(f"**Prescription:** {med}")
                    else:
                        st.warning("No medications prescribed")
                with tab3:
                    if structured_data["advice"]:
                        st.markdown("**Patient Instructions:**")
                        for advice in structured_data["advice"]:
                            st.markdown(f"✅ {advice}")
                    else:
                        st.info("No specific advice provided")
                with tab4:
                    if structured_data["follow_up"].get("date"):
                        st.markdown(f"**Next Appointment:** 📅 {structured_data['follow_up']['date']}")
                    else:
                        st.warning("No follow-up date specified")
                
            if show_raw_text:
                st.subheader("Raw OCR Output")
                st.code(ocr_text)

st.markdown("---")
st.markdown("*Medical data extraction powered by Tesseract OCR and advanced NLP patterns*")
