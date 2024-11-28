# Road Defect Detection

This project uses YOLOv8 to detect defects in road surfaces. It provides a user-friendly web interface built with Streamlit for easy interaction with the model.

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Place the model weights file (`best.pt`) in the `models/` directory.

3. Run the application:
```bash
streamlit run app/main.py
```

## Project Structure

```
road_defect_detection/
├── app/
│   ├── main.py              # Main Streamlit application
│   └── utils/               # Utility functions
├── models/                  # Model weights directory
├── assets/                  # Static assets
│   ├── css/                # Custom CSS
│   └── images/             # Images
└── tests/                  # Test files
```
