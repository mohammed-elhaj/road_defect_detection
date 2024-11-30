from ultralytics import YOLO
import cv2
import numpy as np

class RoadDefectDetector:
    def __init__(self, model_path):
        """Initialize the YOLO model"""
        self.model = YOLO(model_path)
    
    def detect(self, image):
        """Perform detection on an image"""
        results = self.model(image)
        return self._process_results(results[0])
    
    def _process_results(self, result):
        """Process YOLO results into a structured format"""
        return {
            'boxes': [box.xyxy[0].cpu().numpy() for box in result.boxes],
            'scores': [box.conf.cpu().numpy()[0] for box in result.boxes],
            'classes': [box.cls.cpu().numpy()[0] for box in result.boxes],
            'class_names': [result.names[int(box.cls)] for box in result.boxes]
        }
    
    def draw_detections(self, image, detections):
        """Draw boxes and labels on the image"""
        img_copy = image.copy()
        
        for box, score, class_name in zip(
            detections['boxes'], 
            detections['scores'], 
            detections['class_names']
        ):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            cv2.putText(img_copy, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img_copy
