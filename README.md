# Road Defect Detection

A YOLOv8-based system for detecting road defects using Streamlit.

## Project Structure
```
road_defect_detection/
├── app/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── detection.py
│   │   └── visualization.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   └── main.py
├── models/
├── assets/
│   ├── css/
│   │   └── style.css
│   └── images/
├── tests/
│   ├── __init__.py
│   ├── test_detection.py
│   └── test_visualization.py
├── README.md
└── requirements.txt
```

## Setup
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Place model weights in models/best.pt

3. Run the application:
```bash
streamlit run app/main.py
```
