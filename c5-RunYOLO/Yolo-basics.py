from ultralytics import YOLO
import cv2
import os

# Path to YOLO weights and input image
weights_path = '../Yolo-Weights/yolov8n.pt'
image_path = 'images/1.jpg'

# Check if paths exist
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"YOLO weights not found at {weights_path}")
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Input image not found at {image_path}")

# Load YOLO model
model = YOLO(weights_path)

# Run YOLO on the image and display results
results = model(image_path, show=True)

# Wait for key press to close the image display window
cv2.waitKey(0)
cv2.destroyAllWindows()
