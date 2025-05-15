import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import time

# Global variables
model = None
person_tracker = {}  # Track person IDs across frames
behavior_history = {}  # Store behavior history for temporal smoothing

def initialize_model(model_path, device='cpu'):
    """Initialize YOLOv8 model with explicit class names"""
    global model
    try:
        model = YOLO(model_path).to('cpu')
        print(f"Model initialized with classes: {model.names}")
        print("YOLOv8 model loaded successfully")
        return True

    except Exception as e:
        print(f"Failed to load YOLOv8 model: {e}")
        return False

def detect_behavior(image, conf_threshold=0.2):
    """Main function to detect behaviors in an image with improved tracking"""
    global model

    if model is None:
        raise ValueError("Model not initialized. Call initialize_model() first")

    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")

    # Convert image to BGR if it's RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image.copy()
        # Define class-specific thresholds
    class_conf_thresholds = {
        0: conf_threshold /0.9,  # 'hand-raising' - use default threshold
        1: conf_threshold *1.3,  # 'reading' - higher threshold (e.g., 0.6 if default is 0.3)
        2: conf_threshold /1.2 # 'writing' - lower threshold (e.g., 0.15 if default is 0.3)
    }
    detections = []
    results = model(image_bgr, conf=min(class_conf_thresholds.values()) / 2)


    for result in results:
        boxes = result.boxes
        for box in boxes:
            if hasattr(box, 'xyxy'):
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            elif hasattr(box, 'xywh'):
                x, y, w, h = box.xywh.cpu().numpy()[0]
                x1, y1 = x - w / 2, y - h / 2
                x2, y2 = x + w / 2, y + h / 2
            else:
                continue

            bbox = [x1, y1, x2, y2]
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])

            # Apply class-specific threshold
            if conf < class_conf_thresholds.get(cls_id, conf_threshold):
                continue  # Skip this detection if below the class-specific thresh

            # Get class name directly from the model or result
            if hasattr(model, 'names') and cls_id in model.names:
                cls_name = model.names[cls_id]
            elif hasattr(result, 'names') and cls_id in result.names:
                cls_name = result.names[cls_id]
            else:
                # More informative default - helps with debugging
                cls_name = f'class_{cls_id}'
                print(f"Warning: Unknown class ID {cls_id}")

            detection = {
                'bbox': [float(x) for x in bbox],
                'confidence': float(conf),
                'coordinates': {
                    'top_left': (int(x1), int(y1)),
                    'bottom_right': (int(x2), int(y2)),
                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1)
                },
                'class_id': cls_id,
                'class': cls_name,  # Use model's classification directly
                'behavior_confidence': float(conf)  # Use model's confidence
            }

            detections.append(detection)

    # Apply NMS (non-maximum suppression) as before
    if detections:
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        final_dets = []
        used_boxes = []

        for det in detections:
            if det['confidence'] >= conf_threshold / 1.5:
                x1, y1, x2, y2 = det['bbox']
                overlap = False

                for used in used_boxes:
                    xi1 = max(x1, used[0])
                    yi1 = max(y1, used[1])
                    xi2 = min(x2, used[2])
                    yi2 = min(y2, used[3])
                    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    union = (x2 - x1) * (y2 - y1) + (used[2] - used[0]) * (used[3] - used[1]) - inter
                    iou = inter / union if union > 0 else 0

                    if iou > 0.5:
                        overlap = True
                        break

                if not overlap:
                    final_dets.append(det)
                    used_boxes.append([x1, y1, x2, y2])

        if not final_dets and detections:
            final_dets.append(detections[0])

        detections = final_dets

    return detections

def draw_detections(image, detections):
    """Draw detection results on the image with improved visualization"""
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")

    output = image.copy()

    activity_colors = {
        'raised_hand': (0, 255, 0),  # Green
        'reading': (255, 0, 0),  # Blue
        'writing': (0, 0, 255),  # Red
        'class_0': (255, 165, 0),  # Orange (fallback for unknown class 0)
        'class_1': (128, 0, 128),  # Purple (fallback for unknown class 1)
        'class_2': (255, 192, 203)  # Pink (fallback for unknown class 2)
    }

    activity_counts = defaultdict(int)

    # Sort by confidence to draw higher confidence detections on top
    detections = sorted(detections, key=lambda x: x.get('behavior_confidence', x.get('confidence', 0)))

    for det in detections:
        # Get class name - now this should come directly from the model
        cls_name = det.get('class', 'unknown')
        activity_counts[cls_name] += 1

        color = activity_colors.get(cls_name, (255, 255, 255))
        x1, y1, x2, y2 = map(int, det['bbox'])

        # Draw bounding box with thickness based on confidence
        conf = det.get('behavior_confidence', det.get('confidence', 0.5))
        thickness = max(1, min(3, int(conf * 5)))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        # Draw ID if available
        if 'id' in det:
            id_text = f"ID:{det['id']}"
            cv2.putText(output, id_text, (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw label with confidence
        label = f"{cls_name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(output, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


        print(det.keys())
        # Draw center point
        center = det['coordinates']['center']
        cv2.circle(output, center, 3, (0, 0, 255), -1)



    # Add summary
    summary = ", ".join([f"{k}: {v}" for k, v in activity_counts.items()])
    cv2.putText(output, summary, (10, output.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return output


def process_video_stream(video_source=0, output_path=None, model_path='yolov8n.pt'):
    """Process video stream with the behavior detection system"""
    # Initialize model
    if not initialize_model(model_path):
        return False

    # Open video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return False

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup output video if requested
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    previous_detections = None
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames for performance if needed
            if frame_count % 2 != 0:  # Process every other frame
                continue

            # Detect behaviors
            detections = detect_behavior(frame, conf_threshold=0.4, previous_detections=previous_detections)

            # Visualize results
            output_frame = draw_detections(frame, detections)

            # Display result
            cv2.imshow('Behavior Detection', output_frame)

            # Save to output video if requested
            if writer:
                writer.write(output_frame)

            # Update previous detections for next frame
            previous_detections = detections

            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    return True