import os
import cv2
from ultralytics import YOLO
import cvzone

# Load YOLOv8 model
weights_path = "../yolo-weights/yolov8n.pt"  # Adjust the path to your YOLOv8 weights
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"YOLO weights not found at {weights_path}")

model = YOLO(weights_path)

# Define the target class
TARGET_CLASS = "car"  # Class to detect

# Create a directory to store cropped images
cropped_dir = "../c5-RunYOLO/cropped_cars"
os.makedirs(cropped_dir, exist_ok=True)


def process_image(image_path):
    """Process an image to detect cars and crop them."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at {image_path}")

    # Load the image
    img = cv2.imread(image_path)

    # Resize the image while maintaining aspect ratio
    target_width = 1280
    scale = target_width / img.shape[1]
    new_height = int(img.shape[0] * scale)
    resized_img = cv2.resize(img, (target_width, new_height))

    # Run YOLO model on the resized image
    results = model(resized_img, show=False)

    # Counter for cropped images
    crop_count = 0

    # Process detection results
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates and details
            x1, y1, x2, y2 = map(int, box.xyxy)
            w, h = x2 - x1, y2 - y1
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID

            # Get the class name from the model
            class_name = model.names[cls]

            # Filter for the target class
            if class_name == TARGET_CLASS:
                # Crop the detected car
                cropped_car = resized_img[y1:y2, x1:x2]

                # Save the cropped image
                crop_path = os.path.join(cropped_dir, f"car_{crop_count}.jpg")
                cv2.imwrite(crop_path, cropped_car)
                crop_count += 1

                # Draw bounding box and label on the image
                cvzone.cornerRect(resized_img, (x1, y1, w, h), colorR=(0, 255, 0), thickness=2)
                label = f"{class_name} {conf:.2f}"
                cvzone.putTextRect(resized_img, label, (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

    # Display the processed image
    cv2.imshow("Detected Image", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Saved {crop_count} cropped car images to '{cropped_dir}'")


def process_video(video_path):
    """Process a video to detect cars and display live detections."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")

    # Load the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Resize the frame for processing
        target_width = 1280
        scale = target_width / frame.shape[1]
        new_height = int(frame.shape[0] * scale)
        resized_frame = cv2.resize(frame, (target_width, new_height))

        # Run YOLO model on the frame
        results = model(resized_frame, show=False)

        # Process detection results
        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates and details
                x1, y1, x2, y2 = map(int, box.xyxy)
                w, h = x2 - x1, y2 - y1
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class ID

                # Get the class name from the model
                class_name = model.names[cls]

                # Filter for the target class
                if class_name == TARGET_CLASS:
                    # Draw bounding box and label on the frame
                    cvzone.cornerRect(resized_frame, (x1, y1, w, h), colorR=(0, 255, 0), thickness=2)
                    label = f"{class_name} {conf:.2f}"
                    cvzone.putTextRect(resized_frame, label, (max(0, x1), max(35, y1)), scale=0.7, thickness=1)

        # Display the processed frame
        cv2.imshow("Video Detection", resized_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Specify the input files
image_file = "../c5-RunYOLO/images/cars.jpg"
video_file = "../videos/motocar.mp4"

# Process the image
print("Processing image...")
process_image(image_file)

# Process the video
print("Processing video...")
process_video(video_file)
