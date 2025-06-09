import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from collections import defaultdict

# Configuration
CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold for YOLOv8 model
FRAME_SKIP = 1  # Process every frame

# Paths
VIDEO_PATH = "/Users/amansubash/Downloads/ttt.mp4"
OUTPUT_PATH = "/Users/amansubash/Downloads/output_no_helmet.mp4"
MODEL_PATH = "/Users/amansubash/Downloads/h_weights.pt"  # Your local model weights

# Initialize YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)

# Initialize DeepSORT tracker
tracker = DeepSort(
    max_age=30,            # Keep track of disappeared objects for 30 frames
    n_init=3,              # Require 3 detections to confirm track
    nms_max_overlap=0.7,   # Non-maxima suppression threshold
    max_cosine_distance=0.3,  # Threshold for feature similarity
    nn_budget=None,
    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True
)

# Tracking state
tracked_ids = set()

def get_detections(frame):
    """Get no-helmet detections using YOLOv8 model"""
    results = model(frame, conf=CONFIDENCE_THRESHOLD)[0]
    detections = []

    for r in results.boxes.data:
        x1, y1, x2, y2, conf, class_id = r
        if int(class_id) == 0:  # Assuming 0 is the no-helmet class
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            w = x2 - x1
            h = y2 - y1
            
            # Only return high confidence detections
            if conf > CONFIDENCE_THRESHOLD:
                detections.append(([x1, y1, w, h], float(conf), "no-helmet"))
    
    return detections

def draw_info_overlay(frame, active_tracks):
    """Draw count of currently tracked non-helmet riders"""
    height = frame.shape[0]
    width = frame.shape[1]
    
    # Create overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw count
    text = f"Non-Helmet Riders: {active_tracks}"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 
                1.0, (255, 255, 255), 2)
    
    return frame

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_count = 0
    try:
        print("Starting no-helmet detection and tracking...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_SKIP == 0:
                # Get new detections
                detections = get_detections(frame)
                
                # Update tracks
                tracks = tracker.update_tracks(detections, frame=frame)
                
                # Track count for this frame
                active_tracks = 0
                
                # Process and draw tracks
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                        
                    # Get track box
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    
                    # Draw red bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    active_tracks += 1
                    
                    # Add to tracked IDs set
                    tracked_ids.add(track.track_id)
                
                # Update info overlay
                frame = draw_info_overlay(frame, active_tracks)
            
            # Write and display frame
            out.write(frame)
            cv2.imshow("No-Helmet Detection", frame)
            
            # Handle 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessed {frame_count} frames")
        print(f"Total unique tracks: {len(tracked_ids)}")
        print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()