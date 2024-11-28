import streamlit as st
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import tempfile

# Page config
st.set_page_config(
    page_title="Road Defect Detection",
    page_icon="🛣️",
    layout="wide"
)

def load_css():
    css = """
        <style>
            .stApp {
                max-width: 1200px;
                margin: 0 auto;
            }
            .uploadedFile {
                border: 2px dashed #1f77b4;
                padding: 20px;
                border-radius: 10px;
            }
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def load_model():
    if 'model' not in st.session_state:
        with st.spinner("Loading model... This might take a few seconds."):
            st.session_state.model = YOLO('models/best.pt')
    return st.session_state.model

def main():
    load_css()
    
    # Header
    st.title("Road Defect Detection System")
    st.markdown("### Upload an image to detect road defects")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses YOLOv8 to detect road defects. "
        "Upload an image to get started!"
    )
    
    # Model loading
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png']
    )
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Display original image
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # Process and display results
        with col2:
            st.subheader("Detected Defects")
            with st.spinner("Processing..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Run inference
                results = model(tmp_file_path)
                
                # Display results
                res_plotted = results[0].plot()
                st.image(res_plotted, use_column_width=True)
                
                # Display detection information
                boxes = results[0].boxes
                st.write(f"Found {len(boxes)} defects")
                
                if len(boxes) > 0:
                    st.write("Defect Details:")
                    for box in boxes:
                        conf = box.conf[0]
                        cls = box.cls[0]
                        st.write(f"- Type: {results[0].names[int(cls)]} (Confidence: {conf:.2f})")

if __name__ == "__main__":
    main()
