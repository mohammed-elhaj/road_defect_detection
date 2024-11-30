import streamlit as st
from utils.detection import RoadDefectDetector
import cv2
import numpy as np
from PIL import Image
import tempfile

# Initialize detector
@st.cache_resource
def load_detector():
    return RoadDefectDetector(model_path="models/best.pt")

def main():
    st.title("Road Defect Detection")
    
    # Initialize the detector
    try:
        detector = load_detector()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:
            # Create two columns
            col1, col2 = st.columns(2)
            
            # Display original image
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)

            # Process image for detection
            with col2:
                st.subheader("Detected Defects")
                
                # Convert PIL Image to CV2 format
                image_np = np.array(image)
                
                # Perform detection
                with st.spinner("Detecting defects..."):
                    detections = detector.detect(image_np)
                    
                    if len(detections['boxes']) > 0:
                        # Display results
                        result_image = detector.draw_detections(image_np, detections)
                        st.image(result_image, use_column_width=True)
                        
                        # Display detection count
                        st.write(f"Found {len(detections['boxes'])} defects")
                    else:
                        st.write("No defects detected")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
