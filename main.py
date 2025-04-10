import cv2
import numpy as np
from fastapi import FastAPI, Request, BackgroundTasks, File, UploadFile, Form, Query, Depends, Body
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import uvicorn
import torch
import math
import signal
import sys
import threading
import time
import os
import shutil
from pathlib import Path
from typing import Optional, Dict
from collections import defaultdict
from pydantic import BaseModel
from datetime import datetime
import base64

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Create static directory if it doesn't exist
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Flag to control the streaming loop
running = True

# Video source tracking
current_video_path = None

# Selective tracking state
selected_object_id = None

# Webcam state management
webcam_active = False
webcam_thread = None
webcam_source = 0  # Default camera
webcam_frame_buffer = None
webcam_lock = threading.Lock()
webcam_paused = False  # Flag to pause webcam processing
just_resumed = False  # Flag to indicate webcam just resumed from pause

# Standard dimensions for video processing
STD_WIDTH = 1280
STD_HEIGHT = 720

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

# Load the YOLO model
try:
    MODEL_PATH = "final_model_openvino_model/"# Use resolved path
    YOLO_FALLBACK_PATH = "yolov8n.pt" # Ultralytics usually handles caching this

    # Force OpenVINO to use GPU backend
    os.environ["OPENVINO_FORCE_BACKEND"] = "GPU"
    print("Setting OpenVINO to use GPU backend")
    
    print(f"Attempting to load model from: {MODEL_PATH}")
    # Load OpenVINO model directly with YOLO class instead of load_yolo_model function
    
    model = YOLO(MODEL_PATH, task='detect')  # Explicitly specify task as 'detect'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Don't call model.to(device) as it's not supported for OpenVINO format
    # Instead, we'll pass the device in the predict calls
    print(f"OpenVINO model loaded successfully")

    # Warm up the model with a dummy frame to trigger compilation
    print("Warming up model on GPU...")
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    _ = model.predict(dummy_frame, device=device, half=True)
    print("Model warmed up successfully")

except Exception as e:
    print(f"Critical error loading model: {e}")
    print("Using basic YOLO model as fallback")
    model = YOLO("yolov8n.pt")  # Use a standard model as fallback
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)  # This works for PyTorch models

# Store the track history
track_history = defaultdict(list)
box_size_history = defaultdict(list)
direction_history = {}
speed_history = {}

# Add feature-based tracking capabilities
def extract_features(frame, box):
    """Extract appearance features from the object region"""
    x, y, w, h = box
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)
    # Get region of interest and compute histogram features
    roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
    if roi.size == 0:
        return None
    # Convert to HSV for better color consistency
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Calculate histogram for appearance matching
    hist = cv2.calcHist([hsv_roi], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def compare_features(hist1, hist2):
    """Compare two feature histograms for similarity"""
    if hist1 is None or hist2 is None:
        return 0
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

class RobustObjectTracker:
    """Improved object tracker with re-identification capabilities"""
    def __init__(self, object_id, initial_box=None, frame=None):
        self.id = object_id
        self.reference_features = None
        self.last_position = None
        self.last_size = None
        self.velocity = [0, 0]
        self.last_seen = 0
        self.state = "Active"  # Active, Lost, or Recovered
        self.lost_frames = 0
        self.max_lost_frames = 30  # How long to keep predicting position
        self.prediction_only = False
        
        # Initialize with frame and box if provided
        if initial_box is not None and frame is not None:
            self.update_features(frame, initial_box)
            
    def update_features(self, frame, box):
        """Update the reference features for this object"""
        x, y, w, h = box
        self.last_position = (x, y)
        self.last_size = (w, h)
        self.reference_features = extract_features(frame, box)
        
    def predict_position(self):
        """Predict next position based on velocity"""
        if self.last_position is None:
            return None
        x, y = self.last_position
        vx, vy = self.velocity
        return (x + vx, y + vy)
        
    def update_velocity(self, new_position):
        """Update velocity based on position change"""
        if self.last_position is None:
            self.last_position = new_position
            return
        
        x_old, y_old = self.last_position
        x_new, y_new = new_position
        
        # Calculate new velocity with smoothing
        smoothing = 0.7  # Higher value means slower adaptation
        vx = (x_new - x_old) * (1 - smoothing) + self.velocity[0] * smoothing
        vy = (y_new - y_old) * (1 - smoothing) + self.velocity[1] * smoothing
        
        self.velocity = [vx, vy]
        
    def find_best_match(self, frame, boxes, ids):
        """Find the best matching detection for this tracked object"""
        if len(boxes) == 0:
            return None, None, 0
            
        best_match_id = None
        best_match_box = None
        best_score = -1
        
        # Get predicted position if we have velocity
        predicted_pos = self.predict_position()
        
        for i, (box, obj_id) in enumerate(zip(boxes, ids)):
            # Extract features for this detection
            features = extract_features(frame, box)
            
            # Calculate position similarity
            pos_score = 0
            if predicted_pos is not None:
                x, y, w, h = box
                dist = np.sqrt((predicted_pos[0] - x)**2 + (predicted_pos[1] - y)**2)
                pos_score = max(0, 1 - dist / 300)  # Normalize distance (300px max distance)
            
            # Calculate size similarity
            size_score = 0
            if self.last_size is not None:
                w, h = box[2:4]
                w_last, h_last = self.last_size
                size_diff = abs(w/w_last - 1) + abs(h/h_last - 1)
                size_score = max(0, 1 - size_diff / 1.0)  # Normalize size difference
            
            # Calculate appearance similarity
            appear_score = compare_features(self.reference_features, features)
            
            # Combined score with weighted components
            combined_score = (0.4 * pos_score + 
                              0.2 * size_score + 
                              0.4 * appear_score)
            
            if combined_score > best_score:
                best_score = combined_score
                best_match_id = obj_id
                best_match_box = box
        
        threshold = 0.4  # Minimum score required for a valid match
        if best_score < threshold:
            return None, None, 0
            
        return best_match_id, best_match_box, best_score
    
    def get_position(self):
        """Get current position and size (x, y, w, h)"""
        if self.last_position is None or self.last_size is None:
            return (0, 0, 0, 0)  # Default if not initialized
        
        x, y = self.last_position
        w, h = self.last_size
        return (x, y, w, h)
    
    def update_webcam(self, frame):
        """Simple update for webcam mode - just run YOLO to get detections"""
        try:
            # Run YOLO detection on the frame
            results = model.predict(frame, device=device, half=True)
            
            # Process the detections
            if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                # Get all boxes
                boxes = results[0].boxes.xywh.cpu().numpy()  # center_x, center_y, width, height
                confidences = results[0].boxes.conf.cpu().numpy()
                
                # Find the best match based on position overlap with our current box
                if self.last_position is not None and self.last_size is not None:
                    best_box_idx = -1
                    best_overlap = -1
                    
                    curr_x, curr_y = self.last_position
                    curr_w, curr_h = self.last_size
                    curr_area = curr_w * curr_h
                    
                    for i, box in enumerate(boxes):
                        x, y, w, h = box
                        
                        # Simple IoU calculation
                        x1_b1, y1_b1 = curr_x - curr_w/2, curr_y - curr_h/2
                        x2_b1, y2_b1 = curr_x + curr_w/2, curr_y + curr_h/2
                        
                        x1_b2, y1_b2 = x - w/2, y - h/2
                        x2_b2, y2_b2 = x + w/2, y + h/2
                        
                        # Calculate intersection
                        x_left = max(x1_b1, x1_b2)
                        y_top = max(y1_b1, y1_b2)
                        x_right = min(x2_b1, x2_b2)
                        y_bottom = min(y2_b1, y2_b2)
                        
                        if x_right < x_left or y_bottom < y_top:
                            # No overlap
                            continue
                        
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)
                        box_area = w * h
                        union = curr_area + box_area - intersection_area
                        overlap = intersection_area / union if union > 0 else 0
                        
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_box_idx = i
                
                    # If we found a good match, update position
                    if best_box_idx >= 0 and best_overlap > 0.1:
                        x, y, w, h = boxes[best_box_idx]
                        self.update_velocity((x, y))
                        self.last_position = (x, y)
                        self.last_size = (w, h)
                        return True, (x, y, w, h)
            
            # If we got here, no good match was found
            # Try to predict position based on velocity
            predicted_pos = self.predict_position()
            if predicted_pos is not None:
                self.last_position = predicted_pos
                return True, (predicted_pos[0], predicted_pos[1], self.last_size[0], self.last_size[1])
            
            return False, self.get_position()
            
        except Exception as e:
            print(f"Error in robust tracker update: {e}")
            return False, self.get_position()
    
    def update(self, frame, boxes, ids, frame_count):
        """Update tracker with new detections"""
        self.lost_frames += 1
        self.prediction_only = True
        
        # Find best matching detection
        match_id, match_box, match_score = self.find_best_match(frame, boxes, ids)
        
        if match_box is not None:
            # We found a good match
            x, y, w, h = match_box
            
            # Update tracker
            self.last_position = (x, y)
            self.last_size = (w, h)
            self.update_velocity((x, y))
            self.last_seen = frame_count
            self.lost_frames = 0
            self.prediction_only = False
            
            # Update reference features (occasional updates to adapt to appearance changes)
            if frame_count % 10 == 0:
                self.update_features(frame, match_box)
            
            if self.state == "Lost":
                self.state = "Recovered"
            else:
                self.state = "Active"
            
            return match_id, True
        elif self.lost_frames > self.max_lost_frames:
            # Object lost for too long
            self.state = "Lost"
            return None, False
        else:
            # Object temporarily lost, use motion prediction
            self.state = "Lost"
            predicted_pos = self.predict_position()
            if predicted_pos is not None and self.last_size is not None:
                self.last_position = predicted_pos
            return None, True

# Global tracker object for selected object
robust_tracker = None

# Function to predict direction
def predict_direction(points, box_sizes, num_points=5):
    """
    Determine direction of movement based on recent trajectory points
    and changing object size to infer 3D movement.
    """
    if len(points) < num_points or len(box_sizes) < num_points:
        return "Unknown", "Unknown", "Unknown", None

    # Get the recent points and box sizes for direction analysis
    recent_points = points[-num_points:]
    recent_boxes = box_sizes[-num_points:]

    # Calculate the overall movement vector in the 2D plane
    start_x, start_y = recent_points[0]
    end_x, end_y = recent_points[-1]
    dx = end_x - start_x
    dy = start_y - end_y  # Invert y because screen coordinates have y increasing downward

    # Calculate magnitude of horizontal/vertical movement
    magnitude = math.sqrt(dx * dx + dy * dy)

    # If barely moving horizontally/vertically, could still be moving in depth
    if magnitude < 5:  # Threshold for minimum movement in image plane
        horizontal_direction = "Stationary"
        vertical_direction = "Stationary"
        angle = 0
    else:
        # Calculate angle in degrees for 2D plane movement
        angle = math.degrees(math.atan2(dy, dx))

        # Normalize angle to 0-360 range
        if angle < 0:
            angle += 360

        # Determine horizontal direction
        if 112.5 <= angle < 247.5:  # Left half of circle
            horizontal_direction = "Left"
        elif 292.5 <= angle or angle < 67.5:  # Right half of circle
            horizontal_direction = "Right"
        else:
            horizontal_direction = "Stationary"

        # Determine vertical direction
        if 22.5 <= angle < 157.5:  # Upper half of circle
            vertical_direction = "Up"
        elif 202.5 <= angle < 337.5:  # Lower half of circle
            vertical_direction = "Down"
        else:
            vertical_direction = "Stationary"

    # Analyze 3D movement (toward/away from camera) based on changing object size
    start_w, start_h = recent_boxes[0]
    end_w, end_h = recent_boxes[-1]

    # Calculate the ratio of size change
    size_start = start_w * start_h
    size_end = end_w * end_h

    # Avoid division by zero
    if size_start == 0:
        size_start = 1

    size_ratio = size_end / size_start
    size_change_threshold = 0.05  # 5% change threshold

    # Determine if object is moving toward or away based on size change
    if size_ratio > (1 + size_change_threshold):
        depth_direction = "Toward Camera"
    elif size_ratio < (1 - size_change_threshold):
        depth_direction = "Away from Camera"
    else:
        depth_direction = "Same Distance"

    return horizontal_direction, vertical_direction, depth_direction, angle

# Function to calculate speed
def calculate_speed(points, fps, pixels_per_meter, num_points=10):
    """
    Calculate speed based on points trajectory.
    """
    if len(points) < num_points:
        return None, None

    # Use recent points for speed calculation
    recent_points = points[-num_points:]

    # Calculate total distance in pixels
    total_distance = 0
    for i in range(1, len(recent_points)):
        x1, y1 = recent_points[i - 1]
        x2, y2 = recent_points[i]
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += dist

    # Average distance per frame
    distance_per_frame = total_distance / (len(recent_points) - 1)

    # Convert to distance per second
    distance_per_second = distance_per_frame * fps

    # Convert pixels to meters
    distance_meters_per_second = distance_per_second / pixels_per_meter

    # Convert to km/h
    speed_kmh = distance_meters_per_second * 3.6  # 3.6 is the conversion factor from m/s to km/h

    return speed_kmh, distance_per_frame

# Frame counter and FPS settings
frame_count = 0
fps = 30  # Estimated FPS, will be updated when video is loaded
pixels_per_meter = 100  # Calibration for speed calculation

# Fix for OpenCV tracking issues with pyramid sizes
def reset_tracking_cache():
    """Clear tracking history for a fresh start"""
    global track_history, box_size_history, direction_history, speed_history
    track_history = defaultdict(list)
    box_size_history = defaultdict(list)
    direction_history = {}
    speed_history = {}
    
    # Warm up the model with a dummy frame
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    model.predict(dummy_frame, verbose=False, device=device, half=True)

def process_video(filename: str, use_tracking: bool = True, obj_id: Optional[int] = None, start_frame: int = 0):
    """Process the uploaded video and yield frames with detection/tracking"""
    global track_history, box_size_history, direction_history, speed_history, frame_count, fps, running
    global current_video_path, selected_object_id, robust_tracker
    
    # Store current frame for pause functionality
    if not hasattr(process_video, 'current_frame'):
        process_video.current_frame = None
    
    # Store current frame index for resuming
    if not hasattr(process_video, 'current_frame_index'):
        process_video.current_frame_index = 0
    
    # Performance monitoring variables
    frame_times = []
    last_frame_time = time.time()
    processing_start_time = time.time()
    
    # Set selected object ID from parameter if provided
    if obj_id is not None:
        selected_object_id = obj_id
    
    # Reset tracking data for new video or if starting from beginning
    if start_frame == 0:
        track_history = defaultdict(list)
        box_size_history = defaultdict(list)
        direction_history = {}
        speed_history = {}
        frame_count = 0
        robust_tracker = None
    
    # Handle uploaded video
    if not filename:
        print("Error: No filename provided for uploaded video")
        return
        
    video_source = str(UPLOAD_DIR / filename)
    if not os.path.exists(video_source):
        print(f"Error: File not found: {video_source}")
        return
        
    current_video_path = video_source
    
    # Open video file
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    # Get video orientation from metadata
    rotation = 0
    try:
        # Try to get rotation from video metadata
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        print(f"Video rotation metadata: {rotation} degrees")
    except:
        print("No rotation metadata found, assuming 0 degrees")

    # Get actual FPS from video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Fallback if FPS not available
    
    # Get original video dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Ensure minimum dimensions for processing
    width = STD_WIDTH
    height = STD_HEIGHT
    
    # Get total frames in video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {filename}")
    print(f"Original dimensions: {original_width}x{original_height}, Standardized: {width}x{height}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    print(f"Starting from frame: {start_frame}")
    
    # Set starting frame - optimized for resuming
    if start_frame > 0 and start_frame < total_frames:
        print(f"Seeking to frame {start_frame} for resuming playback")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = start_frame
        
        # Adjust frame_count to continue from where we left off
        frame_count = start_frame
    else:
        frame_index = 0
    
    # Maximum history length for tracking
    max_history = 60
    
    # Error counter for tracking failures
    tracking_errors = 0
    max_tracking_errors = 3
    tracking_enabled = use_tracking
    
    # Initialize tracking with dummy frames if needed
    prev_frame = None

    # Optimization: Pre-allocate memory for the frame to reduce allocation overhead
    process_video.last_frame_time = time.time()
    
    # Tracking resumption - when resuming from a pause with tracking enabled,
    # we need to reinitialize tracking to avoid jitter
    if start_frame > 0 and tracking_enabled and selected_object_id is not None:
        reset_tracking_cache()
        print("Reset tracking cache for smooth resumption")

    # Initial features for object appearance modeling
    initial_features = None

    while running:
        frame_start_time = time.time()
        success, frame = cap.read()
        if not success:
            # Loop back to the beginning when video ends
            if os.path.exists(str(current_video_path)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_index = 0
                tracking_errors = 0  # Reset error count
                # Reset tracking history when looping
                track_history = defaultdict(list)
                box_size_history = defaultdict(list)
                direction_history = {}
                speed_history = {}
                prev_frame = None  # Reset previous frame
                success, frame = cap.read()
                if not success:
                    break
            else:
                break

        frame_index += 1
        frame_count += 1
        
        # Apply rotation if needed
        if rotation != 0:
            # Calculate rotation matrix
            center = (frame.shape[1] // 2, frame.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            
            # Apply rotation
            frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))
        
        # Resize frame to standard dimensions to ensure consistent processing
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        
        # Store for optical flow consistency check
        if prev_frame is None:
            prev_frame = frame.copy()
            reset_tracking_cache()
            continue
            
        # Check if frame is similar enough to previous for optical flow
        frame_diff = cv2.absdiff(frame, prev_frame)
        diff_mean = np.mean(frame_diff)
        
        # If frames are too different, reset tracking to avoid optical flow errors
        if diff_mean > 50 and tracking_enabled:  # Threshold for significant change
            print(f"Warning: Large frame difference detected ({diff_mean:.1f}). Resetting tracking.")
            reset_tracking_cache()
        
        # Run detection/tracking on the frame
        try:
            if tracking_enabled and selected_object_id is not None:
                if isinstance(selected_object_id, dict) and 'bbox' in selected_object_id:
                    # Initialize tracking with the selected bounding box
                    # Convert from [x1,y1,x2,y2] to [x1,y1,w,h] if needed
                    bbox = selected_object_id['bbox']
                    if len(bbox) == 4:
                        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                            # It's in [x1,y1,x2,y2] format, convert to [x1,y1,w,h]
                            w = bbox[2] - bbox[0]
                            h = bbox[3] - bbox[1]
                            tracking_box = [bbox[0], bbox[1], w, h]
                        else:
                            # Already in [x1,y1,w,h] format
                            tracking_box = bbox
                    else:
                        tracking_box = bbox  # Use as-is if not in expected format
                    
                    print(f"Initializing tracking with box: {tracking_box}")
                    # Run detection on the current frame
                    if tracking_box is not None:
                        # If we have a tracking box, use it to guide the tracker
                        results = model.track(frame, persist=True, boxes=[tracking_box], device=device, half=True)
                    else:
                        # Otherwise, track all objects
                        results = model.track(frame, persist=True, device=device, half=True)
                    
                    # After first frame, convert selected_object_id to track ID
                    if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                        track_ids = results[0].boxes.id.cpu().numpy()
                        if len(track_ids) > 0:
                            print(f"Assigned track ID: {int(track_ids[0])}")
                            selected_object_id = int(track_ids[0])
                            
                            # Initialize robust tracker with selected object
                            if robust_tracker is None:
                                boxes = results[0].boxes.xywh.cpu().numpy()
                                robust_tracker = RobustObjectTracker(selected_object_id, boxes[0], frame)
                                print(f"Initialized robust tracker for object ID: {selected_object_id}")
                        else:
                            print("No track ID assigned, retrying detection")
                            # If tracking failed, try detection again
                            results = model(frame, device=device, half=True)
                else:
                    # Continue tracking with existing ID
                    results = model.track(frame, persist=True, device=device, half=True)
            else:
                # Use detection only if tracking is disabled or no object selected
                results = model(frame, device=device, half=True)
                
                # When in detection mode, we need to ignore the selected_object_id
                # so that all detected objects are shown
                if not tracking_enabled:
                    selected_object_id = None
            
            # Calculate and log processing time
            processing_time = time.time() - frame_start_time
            frame_times.append(processing_time)
            
            # Store the time since the last frame to regulate frame rate
            current_time = time.time()
            frame_delta = current_time - process_video.last_frame_time
            process_video.last_frame_time = current_time
            
            # Log performance metrics every 30 frames
            if frame_count % 30 == 0:
                avg_time = sum(frame_times[-30:]) / len(frame_times[-30:])
                current_fps = 1.0 / avg_time
                elapsed_time = time.time() - processing_start_time
                print(f"\nPerformance Metrics (Last 30 frames):")
                print(f"Average Processing Time: {avg_time*1000:.2f}ms")
                print(f"Current FPS: {current_fps:.2f}")
                print(f"Frame Index: {frame_index}, Frame Count: {frame_count}")
                print(f"Elapsed Time: {elapsed_time:.2f}s")
                print(f"GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
                print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
                frame_times = []  # Reset for next batch
            
            annotated_frame = frame.copy()
            
            # Create info panel overlay
            info_panel_height = 150
            info_panel_width = 300
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10),
                        (10 + info_panel_width, 10 + info_panel_height), 
                        (220, 220, 220), -1)
            
            # Apply the overlay with transparency
            alpha = 0.7  # Transparency factor
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
            
            # Reset error counter on success
            tracking_errors = 0
            
            if results and results[0].boxes is not None:
                # Get boxes and track IDs if available
                boxes = results[0].boxes.xywh.cpu().numpy() if hasattr(results[0].boxes, 'xywh') else None
                track_ids = None
                if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy()
                
                if boxes is not None:
                    # If no track IDs available, just use box indices
                    if track_ids is None:
                        track_ids = np.arange(len(boxes))
                    
                    # Enhanced tracking with robust tracker
                    found_selected_object = False
                    
                    # If using robust tracking and we have an active tracker
                    if tracking_enabled and selected_object_id is not None and robust_tracker is not None:
                        # Update the robust tracker with new detections
                        matched_id, tracking_ok = robust_tracker.update(frame, boxes, track_ids, frame_count)
                        
                        if tracking_ok:
                            # Extract the most recent position and size for display
                            last_x, last_y = robust_tracker.last_position
                            last_w, last_h = robust_tracker.last_size
                            
                            # Draw the bounding box for the robustly tracked object
                            x1, y1 = int(last_x - last_w / 2), int(last_y - last_h / 2)
                            x2, y2 = int(last_x + last_w / 2), int(last_y + last_h / 2)
                            
                            # Use color based on tracking state
                            if robust_tracker.state == "Active":
                                box_color = (0, 255, 0)  # Green for active tracking
                            elif robust_tracker.state == "Lost":
                                box_color = (0, 0, 255)  # Red for lost but predicting
                            else:  # Recovered
                                box_color = (0, 165, 255)  # Orange for recovered
                            
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                            
                            # Add a label with the object ID
                            track_status = f"ID: {robust_tracker.id}"
                            if robust_tracker.prediction_only:
                                track_status += " (Predicted)"
                            cv2.putText(annotated_frame, track_status, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            
                            # Update tracking history for visualization
                            track_id = robust_tracker.id  # Use the original ID
                            track_history[track_id].append((last_x, last_y))
                            box_size_history[track_id].append((last_w, last_h))
                            
                            # Limit history length
                            if len(track_history[track_id]) > max_history:
                                track_history[track_id].pop(0)
                            if len(box_size_history[track_id]) > max_history:
                                box_size_history[track_id].pop(0)
                            
                            # Draw the tracking line
                            track = track_history[track_id]
                            if len(track) > 1:
                                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.polylines(annotated_frame, [points], isClosed=False,
                                            color=(230, 230, 230), thickness=2)
                            
                            # Calculate direction and speed every 5 frames
                            if frame_count % 5 == 0 and len(track) > 5 and len(box_size_history[track_id]) > 5:
                                horizontal, vertical, depth, angle = predict_direction(
                                    track, box_size_history[track_id])
                                
                                # Calculate speed
                                speed_kmh, px_per_frame = calculate_speed(track, fps, pixels_per_meter)
                                
                                # Store direction and speed data
                                direction_history[track_id] = {
                                    'horizontal': horizontal,
                                    'vertical': vertical,
                                    'depth': depth,
                                    'angle': angle
                                }
                                
                                if speed_kmh is not None:
                                    speed_history[track_id] = {
                                        'kmh': speed_kmh,
                                        'px_per_frame': px_per_frame
                                    }
                            
                            # Draw direction arrow
                            if track_id in direction_history:
                                dir_data = direction_history[track_id]
                                if (dir_data['horizontal'] != "Stationary" or 
                                    dir_data['vertical'] != "Stationary"):
                                    
                                    angle = dir_data['angle']
                                    arrow_start = (int(track[-1][0]), int(track[-1][1]))
                                    
                                    if angle is not None:
                                        arrow_length = 40
                                        end_x = arrow_start[0] + arrow_length * math.cos(math.radians(angle))
                                        end_y = arrow_start[1] - arrow_length * math.sin(math.radians(angle))
                                        arrow_end = (int(end_x), int(end_y))
                                        
                                        # Draw direction arrow
                                        cv2.arrowedLine(annotated_frame, arrow_start, arrow_end,
                                                    (0, 165, 255), 2, tipLength=0.3)
                            
                            # Draw depth movement arrow
                            if track_id in direction_history:
                                depth_dir = direction_history[track_id]['depth']
                                if depth_dir != "Same Distance":
                                    center_x, center_y = int(last_x), int(last_y)
                                    
                                    if depth_dir == "Toward Camera":
                                        cv2.arrowedLine(annotated_frame,
                                                    (center_x, center_y),
                                                    (center_x, center_y - 40),
                                                    (255, 0, 255), 2, tipLength=0.3)
                                    else:  # Away from Camera
                                        cv2.arrowedLine(annotated_frame,
                                                    (center_x, center_y),
                                                    (center_x, center_y + 40),
                                                    (255, 0, 255), 2, tipLength=0.3)
                            
                            # Display information in overlay
                            y_position = 35
                            
                            # Display object ID and tracking state
                            cv2.putText(annotated_frame, f"Object ID: {track_id} ({robust_tracker.state})", 
                                      (20, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            y_position += 20
                            
                            # Display direction info
                            if track_id in direction_history:
                                dir_data = direction_history[track_id]
                                direction_text = f"Direction: {dir_data['horizontal']} | {dir_data['vertical']}"
                                cv2.putText(annotated_frame, direction_text, (20, y_position),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                                y_position += 20
                                
                                depth_text = f"Depth: {dir_data['depth']}"
                                cv2.putText(annotated_frame, depth_text, (20, y_position),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                                y_position += 20
                            
                            # Display speed info
                            if track_id in speed_history:
                                speed_data = speed_history[track_id]
                                speed_text = f"Speed: {speed_data['kmh']:.1f} km/h"
                                cv2.putText(annotated_frame, speed_text, (20, y_position),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                                y_position += 20
                            
                            # Mark the selected object as found
                            found_selected_object = True
                    
                    # Process all other objects (or all if no robust tracking)
                    if not found_selected_object or not tracking_enabled:
                        for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                            # Skip if we're doing selective tracking and this isn't our object
                            if tracking_enabled and selected_object_id is not None and track_id != selected_object_id:
                                continue
                                
                            # Initialize robust tracker if we found our selected object
                            if tracking_enabled and selected_object_id is not None and track_id == selected_object_id:
                                if robust_tracker is None:
                                    robust_tracker = RobustObjectTracker(selected_object_id, box, frame)
                                    print(f"Initialized robust tracker for object ID: {selected_object_id}")
                                else:
                                    # Just update tracker with new position
                                    robust_tracker.update_features(frame, box)
                            
                            # Extract box coordinates
                            x, y, w, h = box
                            x, y, w, h = float(x), float(y), float(w), float(h)
                            
                            # Draw bounding box
                            x1, y1 = int(x - w / 2), int(y - h / 2)
                            x2, y2 = int(x + w / 2), int(y + h / 2)
                            
                            # Use green color for all objects or a different color for selected object
                            box_color = (0, 255, 0)  # Default green
                            if selected_object_id is not None and track_id == selected_object_id:
                                box_color = (0, 165, 255)  # Orange for selected object
                            
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                            
                            # Add a label with the object ID
                            label = f"ID: {track_id}"
                            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            
                            # Track position history
                            track = track_history[track_id]
                            track.append((x, y))  # center point
                            
                            # Track box size history
                            box_sizes = box_size_history[track_id]
                            box_sizes.append((w, h))
                            
                            # Limit history length
                            if len(track) > max_history:
                                track.pop(0)
                            if len(box_sizes) > max_history:
                                box_sizes.pop(0)
                            
                            # Only show tracking history and direction info when tracking is enabled
                            if tracking_enabled:
                                # Draw the tracking line
                                if len(track) > 1:
                                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                                    cv2.polylines(annotated_frame, [points], isClosed=False,
                                                color=(230, 230, 230), thickness=2)
                                
                                # Calculate direction and speed every 5 frames
                                if frame_count % 5 == 0 and len(track) > 5 and len(box_sizes) > 5:
                                    horizontal, vertical, depth, angle = predict_direction(track, box_sizes)
                                    
                                    # Calculate speed
                                    speed_kmh, px_per_frame = calculate_speed(track, fps, pixels_per_meter)
                                    
                                    # Store direction and speed data
                                    direction_history[track_id] = {
                                        'horizontal': horizontal,
                                        'vertical': vertical,
                                        'depth': depth,
                                        'angle': angle
                                    }
                                    
                                    if speed_kmh is not None:
                                        speed_history[track_id] = {
                                            'kmh': speed_kmh,
                                            'px_per_frame': px_per_frame
                                        }
                                
                                # Draw direction arrow
                                if track_id in direction_history:
                                    dir_data = direction_history[track_id]
                                    if (dir_data['horizontal'] != "Stationary" or 
                                        dir_data['vertical'] != "Stationary"):
                                        
                                        angle = dir_data['angle']
                                        arrow_start = (int(track[-1][0]), int(track[-1][1]))
                                        
                                        if angle is not None:
                                            arrow_length = 40
                                            end_x = arrow_start[0] + arrow_length * math.cos(math.radians(angle))
                                            end_y = arrow_start[1] - arrow_length * math.sin(math.radians(angle))
                                            arrow_end = (int(end_x), int(end_y))
                                            
                                            # Draw direction arrow
                                            cv2.arrowedLine(annotated_frame, arrow_start, arrow_end,
                                                        (0, 165, 255), 2, tipLength=0.3)
                                
                                # Draw depth movement arrow
                                if track_id in direction_history:
                                    depth_dir = direction_history[track_id]['depth']
                                    if depth_dir != "Same Distance":
                                        center_x, center_y = int(x), int(y)
                                        
                                        if depth_dir == "Toward Camera":
                                            cv2.arrowedLine(annotated_frame,
                                                        (center_x, center_y),
                                                        (center_x, center_y - 40),
                                                        (255, 0, 255), 2, tipLength=0.3)
                                        else:  # Away from Camera
                                            cv2.arrowedLine(annotated_frame,
                                                        (center_x, center_y),
                                                        (center_x, center_y + 40),
                                                        (255, 0, 255), 2, tipLength=0.3)
                                
                                # Display information in overlay (only if not using robust tracker)
                                if not (tracking_enabled and robust_tracker is not None):
                                    y_position = 35
                                    
                                    # Display object ID
                                    cv2.putText(annotated_frame, f"Object ID: {track_id}", (20, y_position),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                                    y_position += 20
                                    
                                    # Display direction info
                                    if track_id in direction_history:
                                        dir_data = direction_history[track_id]
                                        direction_text = f"Direction: {dir_data['horizontal']} | {dir_data['vertical']}"
                                        cv2.putText(annotated_frame, direction_text, (20, y_position),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                                        y_position += 20
                                        
                                        depth_text = f"Depth: {dir_data['depth']}"
                                        cv2.putText(annotated_frame, depth_text, (20, y_position),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                                        y_position += 20
                                    
                                    # Display speed info
                                    if track_id in speed_history:
                                        speed_data = speed_history[track_id]
                                        speed_text = f"Speed: {speed_data['kmh']:.1f} km/h"
                                        cv2.putText(annotated_frame, speed_text, (20, y_position),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            else:
                                # In detection mode, just display the class name (or object ID)
                                y_position = 35
                                cv2.putText(annotated_frame, f"Object ID: {track_id}", (20, y_position),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
        except Exception as e:
            # If tracking fails, increase error counter and possibly disable tracking
            tracking_errors += 1
            print(f"Error in model inference: {e}")
            
            if tracking_errors >= max_tracking_errors and tracking_enabled:
                print(f"Too many tracking errors, switching to detection only mode")
                tracking_enabled = False
                reset_tracking_cache()
            
            annotated_frame = frame.copy()
            # Add error message to the frame
            cv2.putText(
                annotated_frame,
                f"Model error: {str(e)[:50]}...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

        # Display source and tracking information
        source_text = f"Source: {os.path.basename(current_video_path)}"
        if tracking_enabled and selected_object_id is not None:
            if robust_tracker is not None:
                mode_text = f"Mode: Enhanced Tracking ({robust_tracker.state})"
            else:
                mode_text = "Mode: Tracking"
        else:
            mode_text = "Mode: Detection"
        frame_text = f"Frame: {frame_index}"
        
        cv2.putText(annotated_frame, source_text, (20, 130),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(annotated_frame, mode_text, (20, 150),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(annotated_frame, frame_text, (20, 170),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Add selective tracking status info
        if selected_object_id is not None:
            select_text = f"Tracking Object ID: {selected_object_id}"
            cv2.putText(annotated_frame, select_text, (20, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Update previous frame for next iteration
        prev_frame = frame.copy()
        
        # Add a controlled delay to simulate real-time speed
        target_frame_time = 1.0 / fps  # Target time per frame based on video fps
        elapsed = time.time() - frame_start_time  # How long processing took
        
        # Only sleep if processing was faster than target frame time
        if elapsed < target_frame_time:
            sleep_time = target_frame_time - elapsed
            # Apply a dynamic adjustment to smooth out playback
            sleep_time = min(sleep_time, 0.1)  # Cap sleep time to avoid excessive delays
            time.sleep(sleep_time)
        
        # Store the current frame and index for pause functionality
        process_video.current_frame = annotated_frame.copy()
        process_video.current_frame_index = frame_index

        # Encode the frame with higher quality
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()
    process_video.current_frame = None  # Clear the stored frame when video ends

@app.get("/")
async def index(request: Request):
    # Render the HTML page that displays the video stream
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed(
    filename: str = Query(..., description="Filename for uploaded video"),
    tracking: bool = Query(False, description="Enable object tracking"),
    object_id: Optional[int] = Query(None, description="ID of specific object to track"),
    start_frame: int = Query(0, description="Starting frame index")
):
    """Stream video from uploaded file"""
    print(f"Video feed request: filename={filename}, tracking={tracking}, object_id={object_id}, start_frame={start_frame}")
    return StreamingResponse(
        process_video(filename, use_tracking=tracking, obj_id=object_id, start_frame=start_frame),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/upload_video")
async def upload_video(video: UploadFile = File(...)):
    """Upload a video file for processing"""
    global track_history, box_size_history, direction_history, speed_history, selected_object_id
    
    try:
        # Validate file type
        if not video.content_type.startswith("video/"):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Uploaded file is not a video"}
            )
        
        # Generate a safe filename
        filename = f"{int(time.time())}_{video.filename}"
        file_path = UPLOAD_DIR / filename
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Clean up previous videos
        cleanup_previous_videos(file_path)
        
        # Reset tracking data and caches
        reset_tracking_data()
        
        # Return success response with the filename for client-side use
        return {"success": True, "filename": filename}
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

def cleanup_previous_videos(current_file_path):
    """Clean up all videos except the current one"""
    try:
        current_file = os.path.basename(str(current_file_path))
        # Check all files in the upload directory
        for file in UPLOAD_DIR.glob("*"):
            if file.is_file() and str(file.name) != current_file:
                try:
                    print(f"Removing previous video: {file}")
                    file.unlink()
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")
        
        # Report remaining files (should be just the current one)
        remaining = list(UPLOAD_DIR.glob("*"))
        print(f"Remaining files in upload directory: {len(remaining)}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def reset_tracking_data():
    """Reset all tracking data and caches"""
    global track_history, box_size_history, direction_history, speed_history, selected_object_id
    global frame_count, fps, robust_tracker
    
    print("Resetting tracking data and caches")
    
    # Reset tracking history
    track_history = defaultdict(list)
    box_size_history = defaultdict(list)
    direction_history = {}
    speed_history = {}
    
    # Reset frame counter
    frame_count = 0
    
    # Clear selected object
    selected_object_id = None
    
    # Clear robust tracker
    robust_tracker = None
    
    # Reset any stored frames in the process_video function
    if hasattr(process_video, 'current_frame'):
        process_video.current_frame = None
    
    if hasattr(process_video, 'current_frame_index'):
        process_video.current_frame_index = 0
    
    # Force garbage collection to free memory
    import gc
    gc.collect()
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared GPU cache")
    
    print("All tracking data and caches have been reset")

@app.post("/select_object")
async def select_object(bbox: Dict = Body(...)):
    """Set the object to track based on initial bounding box"""
    global selected_object_id, model, current_video_path, robust_tracker, webcam_frame_buffer
    
    try:
        # Convert relative coordinates to absolute
        rel_bbox = BoundingBox(**bbox['bbox'])
        
        # Convert to absolute pixel coordinates
        abs_x1 = int(rel_bbox.x * STD_WIDTH)
        abs_y1 = int(rel_bbox.y * STD_HEIGHT)
        abs_x2 = int((rel_bbox.x + rel_bbox.width) * STD_WIDTH)
        abs_y2 = int((rel_bbox.y + rel_bbox.height) * STD_HEIGHT)
        
        # Get current frame - either from webcam buffer or video file
        frame = None
        
        # First check if we have a webcam frame buffer
        if webcam_active and webcam_frame_buffer is not None:
            print("Using webcam frame for object selection")
            frame = webcam_frame_buffer.copy()
            frame = cv2.resize(frame, (STD_WIDTH, STD_HEIGHT), interpolation=cv2.INTER_AREA)
        # Otherwise try to get frame from video file
        elif current_video_path and os.path.exists(current_video_path):
            print("Using video frame for object selection")
            cap = cv2.VideoCapture(current_video_path)
            success, frame = cap.read()
            cap.release()
            
            if not success:
                return {"success": False, "error": "Could not read video frame"}
            
            # Resize frame to standard dimensions
            frame = cv2.resize(frame, (STD_WIDTH, STD_HEIGHT), interpolation=cv2.INTER_AREA)
        else:
            return {"success": False, "error": "No active video or webcam stream loaded"}
        
        # Run detector on current frame
        try:
            results = model(frame)
            print(f"Detection found {len(results[0].boxes.data)} objects")
            
            if not hasattr(results[0], 'boxes') or len(results[0].boxes.data) == 0:
                print("No objects detected in frame")
                # If no objects detected, use the user's selection directly
                selected_object_id = {
                    'bbox': [abs_x1, abs_y1, abs_x2, abs_y2]
                }
                
                # Create bounding box for tracker in xywh format
                w = abs_x2 - abs_x1
                h = abs_y2 - abs_y1
                center_x = abs_x1 + w/2
                center_y = abs_y1 + h/2
                
                # Initialize robust tracker with manual selection
                robust_tracker = RobustObjectTracker(1, [center_x, center_y, w, h], frame)
                print("Initialized robust tracker with manual selection")
                
                return {"success": True, "selected_id": 1, "method": "manual_selection"}
            
            # Get all detected boxes
            boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] format
            confs = results[0].boxes.conf.cpu().numpy()
            
            # Calculate IoU (Intersection over Union) with user's selection
            best_iou = 0
            best_idx = -1
            user_box = [abs_x1, abs_y1, abs_x2, abs_y2]
            user_box_area = (abs_x2 - abs_x1) * (abs_y2 - abs_y1)
            
            for i, box in enumerate(boxes):
                # Calculate intersection
                ix1 = max(box[0], user_box[0])
                iy1 = max(box[1], user_box[1])
                ix2 = min(box[2], user_box[2])
                iy2 = min(box[3], user_box[3])
                
                if ix2 < ix1 or iy2 < iy1:
                    # No intersection
                    continue
                
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                union_area = box_area + user_box_area - intersection_area
                iou = intersection_area / union_area
                
                # If we have a better match, update
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            # If we found a good match
            if best_idx >= 0 and best_iou > 0.1:  # Threshold can be adjusted
                print(f"Found matching object with IoU: {best_iou:.2f}")
                best_box = boxes[best_idx]
                
                # Create bounding box for tracker in xywh format
                w = best_box[2] - best_box[0]
                h = best_box[3] - best_box[1]
                center_x = best_box[0] + w/2
                center_y = best_box[1] + h/2
                
                # Use the detected box for improved tracking
                selected_object_id = {
                    'bbox': [float(best_box[0]), float(best_box[1]), 
                            float(best_box[2]), float(best_box[3])]
                }
                
                # Initialize robust tracker with detected box
                robust_tracker = RobustObjectTracker(int(best_idx), 
                                                    [center_x, center_y, w, h], 
                                                    frame)
                print(f"Initialized robust tracker with detection-refined box")
                
                return {"success": True, "selected_id": int(best_idx), 
                        "method": "detector_refined", "confidence": float(confs[best_idx]),
                        "iou": float(best_iou)}
            else:
                print("No matching objects found, using manual selection")
                # Fallback to user's selection
                selected_object_id = {
                    'bbox': [abs_x1, abs_y1, abs_x2, abs_y2]
                }
                
                # Create bounding box for tracker in xywh format
                w = abs_x2 - abs_x1
                h = abs_y2 - abs_y1
                center_x = abs_x1 + w/2
                center_y = abs_y1 + h/2
                
                # Initialize robust tracker with manual selection
                robust_tracker = RobustObjectTracker(1, [center_x, center_y, w, h], frame)
                print("Initialized robust tracker with manual selection")
                
                return {"success": True, "selected_id": 1, "method": "manual_selection"}
            
        except Exception as detect_err:
            print(f"Error running detector: {detect_err}")
            # Fallback to user's selection on detection error
            selected_object_id = {
                'bbox': [abs_x1, abs_y1, abs_x2, abs_y2]
            }
            
            # Create bounding box for tracker in xywh format
            w = abs_x2 - abs_x1
            h = abs_y2 - abs_y1
            center_x = abs_x1 + w/2
            center_y = abs_y1 + h/2
            
            # Initialize robust tracker with manual selection even after error
            try:
                robust_tracker = RobustObjectTracker(1, [center_x, center_y, w, h], frame)
                print("Initialized robust tracker with manual selection after detection error")
            except Exception as tracker_err:
                print(f"Failed to initialize tracker: {tracker_err}")
            
            return {"success": True, "selected_id": 1, "method": "manual_selection"}
            
    except Exception as e:
        print(f"Error in select_object: {e}")
        return {"success": False, "error": str(e)}

@app.post("/clear_selection")
async def clear_selection():
    """Clear the selected object ID"""
    global selected_object_id, robust_tracker
    
    selected_object_id = None
    robust_tracker = None
    print("Cleared object selection and robust tracker")
    return {"success": True}

@app.get("/shutdown")
async def shutdown_server(background_tasks: BackgroundTasks):
    """Endpoint to shut down the server"""
    global running
    running = False
    
    def shutdown():
        time.sleep(1)  # Give a second for response to return
        # Send SIGINT to the current process
        pid = os.getpid()
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.GenerateConsoleCtrlEvent(0, 0)  # Send Ctrl+C
        else:
            os.kill(pid, signal.SIGINT)
    
    background_tasks.add_task(shutdown)
    return JSONResponse({"message": "Server shutting down..."})

@app.on_event("startup")
async def startup_event():
    """Print server information and startup instructions"""
    print("\n" + "="*50)
    print("GPU Real-time Object Tracking Server")
    print("="*50)
    
    # Enhanced GPU diagnostics
    print("\nDetailed GPU Diagnostics:")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        # Verify model is on GPU - safely handle both PyTorch and OpenVINO models
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                print(f"Model Device: {next(model.model.parameters()).device}")
            else:
                print(f"Model Device: {device} (OpenVINO model)")
        except (AttributeError, TypeError):
            print(f"Model Device: {device} (OpenVINO model)")
    else:
        print("CUDA Available: No (Running in CPU mode - performance will be limited)")
        print("Warning: Model is running on CPU, which will be significantly slower")
    
    print(f"\nPython Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"OpenCV Version: {cv2.__version__}")
    
    print("\nServer running on http://0.0.0.0:8000")
    print("To stop the server:")
    print("1. Press 'q' in the terminal")
    print("2. Or visit http://localhost:8000/shutdown in your browser")
    print("3. Or use Ctrl+C in the terminal")
    print("\nTo clear data without stopping the server:")
    print("1. Use the 'Clear Data' button in the web interface")
    print("2. Or visit http://localhost:8000/clear_data with a POST request")
    print("="*50 + "\n")
    
    # Start a thread to listen for 'q' in the terminal
    def key_capture_thread():
        global running
        while running:
            try:
                if input().lower() == 'q':
                    print("Shutting down server...")
                    running = False
                    # Send shutdown signal
                    pid = os.getpid()
                    if sys.platform == 'win32':
                        import ctypes
                        kernel32 = ctypes.windll.kernel32
                        kernel32.GenerateConsoleCtrlEvent(0, 0)  # Send Ctrl+C
                    else:
                        os.kill(pid, signal.SIGINT)
                    break
            except (KeyboardInterrupt, EOFError):
                break
    
    threading.Thread(target=key_capture_thread, daemon=True).start()

# Clean up uploaded files when the app shuts down
@app.on_event("shutdown")
async def cleanup():
    """Clean up temporary files and uploaded videos"""
    print("Cleaning up uploaded files...")
    if UPLOAD_DIR.exists():
        # Count videos to cleanup
        video_files = list(UPLOAD_DIR.glob("*"))
        print(f"Found {len(video_files)} files to clean up")
        
        # Delete each file
        for file in video_files:
            try:
                print(f"Deleting {file}")
                file.unlink(missing_ok=True)  # Python 3.8+ supports missing_ok
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        
        # Verify cleanup
        remaining = list(UPLOAD_DIR.glob("*"))
        if remaining:
            print(f"Warning: {len(remaining)} files could not be deleted")
        else:
            print("All uploaded files cleaned up successfully")

@app.get("/video_frame")
async def video_frame(
    filename: str = Query(..., description="Filename for uploaded video"),
    frame: str = Query("first", description="Which frame to return (first/current)")
):
    """Return a single frame from the video"""
    try:
        video_path = str(UPLOAD_DIR / filename)
        if not os.path.exists(video_path):
            return JSONResponse(
                status_code=404,
                content={"error": "Video file not found"}
            )
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return JSONResponse(
                status_code=500,
                content={"error": "Could not open video"}
            )

        # Get video orientation from metadata
        rotation = 0
        try:
            # Try to get rotation from video metadata
            rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
            print(f"Video frame rotation metadata: {rotation} degrees")
        except:
            # If metadata is not available, try to detect rotation from dimensions
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width < height:  # Portrait video
                rotation = 90
            print(f"No rotation metadata found, detected rotation: {rotation} degrees")
        
        # Get frame based on request
        if frame == "first":
            # Always get first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif frame == "current" and hasattr(process_video, 'current_frame'):
            # Get the last processed frame if available
            frame_data = process_video.current_frame
            if frame_data is not None:
                print(f"Returning cached current frame at index {process_video.current_frame_index}")
                _, buffer = cv2.imencode('.jpg', frame_data, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_bytes = buffer.tobytes()
                cap.release()
                return Response(content=frame_bytes, media_type="image/jpeg")
        
        # Read and return the frame
        success, frame = cap.read()
        cap.release()
        
        if not success:
            return JSONResponse(
                status_code=500,
                content={"error": "Could not read frame"}
            )

        # Apply rotation if needed
        if rotation != 0:
            # Calculate rotation matrix
            center = (frame.shape[1] // 2, frame.shape[0] // 2)
            # Invert rotation for display
            rotation_matrix = cv2.getRotationMatrix2D(center, -rotation, 1.0)
            
            # Get rotated dimensions
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_width = int((frame.shape[0] * sin) + (frame.shape[1] * cos))
            new_height = int((frame.shape[0] * cos) + (frame.shape[1] * sin))
            
            # Adjust translation
            rotation_matrix[0, 2] += (new_width / 2) - (frame.shape[1] / 2)
            rotation_matrix[1, 2] += (new_height / 2) - (frame.shape[0] / 2)
            
            # Apply rotation with adjusted dimensions
            frame = cv2.warpAffine(frame, rotation_matrix, (new_width, new_height))
        
        # Resize frame to standard dimensions while maintaining aspect ratio
        target_height = STD_HEIGHT
        target_width = STD_WIDTH
        
        # Calculate aspect ratio
        aspect_ratio = frame.shape[1] / frame.shape[0]
        target_aspect = target_width / target_height
        
        if aspect_ratio > target_aspect:
            # Image is wider than target
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Image is taller than target
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        # Resize frame
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create black canvas of target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate position to center the frame
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        # Place frame in center of canvas
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = frame
        
        # Convert frame to JPEG with high quality
        _, buffer = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_bytes = buffer.tobytes()
        
        return Response(content=frame_bytes, media_type="image/jpeg")
        
    except Exception as e:
        print(f"Error in video_frame: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/current_frame_index")
async def current_frame_index():
    """Return the current frame index for resuming videos"""
    if hasattr(process_video, 'current_frame_index'):
        return {"success": True, "frame_index": process_video.current_frame_index}
    else:
        return {"success": False, "frame_index": 0}

def get_webcam_frame():
    """Get the latest frame from webcam buffer"""
    global webcam_frame_buffer
    with webcam_lock:
        if webcam_frame_buffer is not None:
            return webcam_frame_buffer.copy()
        return None

def webcam_stream_thread():
    """Background thread to capture webcam frames"""
    global webcam_active, webcam_frame_buffer, webcam_source, webcam_paused
    
    print(f"Starting webcam thread with source: {webcam_source}")
    
    # Special case for browser camera
    if webcam_source == "browser":
        print("Browser camera selected. Waiting for browser to send frames...")
        webcam_active = True
        return  # Exit thread - frames will be sent from browser
    
    # Special case for demo mode or no camera
    if webcam_source == "no_cam":
        print("No camera available - entering placeholder mode")
        webcam_active = True
        return  # Exit thread - no actual camera processing needed
    
    try:
        # Try to interpret as a local camera index
        try:
            camera_idx = int(webcam_source)
            print(f"Attempting to open local camera with index: {camera_idx}")
            
            # Try different backend options if default fails
            backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]
            
            # Try each backend until one works
            cap = None
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(camera_idx, backend)
                    if cap.isOpened():
                        # Successfully opened
                        print(f"Successfully opened camera {camera_idx} with backend {backend}")
                        break
                except Exception as e:
                    print(f"Failed to open camera with backend {backend}: {e}")
                    if cap is not None:
                        cap.release()
                        cap = None
        except ValueError:
            print(f"Error: Invalid camera source '{webcam_source}'")
            webcam_active = False
            return
            
        # Check if we have a valid capture object    
        if cap is None or not cap.isOpened():
            print(f"Error: Could not open webcam source {webcam_source} with any backend")
            webcam_active = False
            return
            
        # Test if we can actually read frames
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print(f"Error: Camera opened but could not read frames")
            cap.release()
            webcam_active = False
            return
            
        # Set resolution - only if camera supports it
        original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Original camera resolution: {original_width}x{original_height}")
        
        # Try to set resolution, but don't fail if it doesn't work
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, STD_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STD_HEIGHT)
        
        # Check if resolution was actually set
        new_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        new_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"New camera resolution: {new_width}x{new_height}")
        
        # Main frame capture loop
        frame_count = 0
        error_count = 0
        max_errors = 5  # Allow up to 5 consecutive errors before giving up
        
        while webcam_active:
            try:
                if not webcam_paused:
                    success, frame = cap.read()
                    if not success:
                        error_count += 1
                        print(f"Error reading from webcam (error {error_count}/{max_errors})")
                        if error_count >= max_errors:
                            print("Too many consecutive errors, stopping webcam thread")
                            break
                        time.sleep(0.1)  # Short delay before retry
                        continue
                    
                    # Reset error count on successful frame read
                    error_count = 0
                    frame_count += 1
                    
                    # Update frame buffer with latest frame
                    with webcam_lock:
                        webcam_frame_buffer = frame.copy()
                
                # Limit frame rate to avoid excessive CPU usage
                time.sleep(0.03)  # ~30fps
                
                # Periodically log that the webcam is still running
                if frame_count % 100 == 0 and not webcam_paused:
                    print(f"Webcam still running, captured {frame_count} frames")
                    
            except Exception as e:
                error_count += 1
                print(f"Exception in webcam thread: {e}")
                if error_count >= max_errors:
                    print("Too many consecutive errors, stopping webcam thread")
                    break
                time.sleep(0.1)  # Short delay before retry
        
        # Properly release the camera
        print("Webcam thread stopping, releasing camera")
        if cap is not None:
            cap.release()
        
    except Exception as e:
        print(f"Critical error in webcam thread: {e}")
        import traceback
        traceback.print_exc()
        webcam_active = False
    
    print("Webcam thread stopped")

@app.post("/start_webcam")
async def start_webcam(camera_id: str = Form("0")):
    """Start webcam stream with specified camera"""
    global webcam_active, webcam_thread, webcam_source, selected_object_id, webcam_frame_buffer
    
    # Stop existing stream if running
    if webcam_active and webcam_thread and webcam_thread.is_alive():
        print("Stopping existing webcam thread")
        webcam_active = False
        webcam_thread.join(timeout=2.0)
    
    # Reset tracking data
    reset_tracking_data()
    
    # Make sure webcam buffer is cleared
    with webcam_lock:
        webcam_frame_buffer = None
    
    # Start new stream
    try:
        print(f"Starting webcam with camera ID: {camera_id}")
        webcam_active = True
        webcam_source = camera_id
        webcam_thread = threading.Thread(target=webcam_stream_thread)
        webcam_thread.daemon = True
        webcam_thread.start()
        
        # Allow some time for the webcam to initialize
        wait_time = 0
        max_wait = 3.0  # Wait up to 3 seconds
        
        while wait_time < max_wait:
            with webcam_lock:
                if webcam_frame_buffer is not None:
                    print("Webcam initialized successfully")
                    return {"success": True, "message": f"Started webcam stream with camera {camera_id}"}
            
            # Check if thread is still alive
            if not webcam_thread.is_alive():
                print("Webcam thread died during initialization")
                return {"success": False, "message": "Failed to initialize webcam"}
                
            # Wait a bit and try again
            time.sleep(0.1)
            wait_time += 0.1
        
        # If we get here, we timed out waiting for the first frame
        if webcam_active and webcam_thread.is_alive():
            print("Webcam thread is alive but no frames received yet")
            return {"success": True, "message": "Webcam started but waiting for first frame"}
        else:
            print("Webcam failed to initialize in time")
            webcam_active = False
            return {"success": False, "message": "Failed to initialize webcam"}
            
    except Exception as e:
        print(f"Error starting webcam: {e}")
        webcam_active = False
        return {"success": False, "message": f"Error starting webcam: {str(e)}"}

@app.post("/stop_webcam")
async def stop_webcam():
    """Stop webcam stream"""
    global webcam_active, webcam_thread
    
    if webcam_active and webcam_thread and webcam_thread.is_alive():
        webcam_active = False
        webcam_thread.join(timeout=2.0)
        return {"success": True, "message": "Webcam stream stopped"}
    
    return {"success": False, "message": "No active webcam stream to stop"}

@app.get("/webcam_feed")
async def webcam_feed(
    tracking: bool = Query(False, description="Enable object tracking"),
    object_id: Optional[int] = Query(None, description="ID of specific object to track")
):
    """Stream video from webcam with object detection/tracking"""
    return StreamingResponse(
        process_webcam_stream(use_tracking=tracking, obj_id=object_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

def shutdown_webcam():
    """Shutdown webcam stream gracefully"""
    global webcam_active, webcam_thread
    
    if webcam_active and webcam_thread and webcam_thread.is_alive():
        print("Shutting down webcam...")
        webcam_active = False
        webcam_thread.join(timeout=2.0)
        print("Webcam shutdown complete")

# Set up clean shutdown handler
def signal_handler(sig, frame):
    global running
    print('Shutting down server...')
    running = False
    
    # Shutdown the webcam if active
    shutdown_webcam()
    
    # Give time for threads to exit
    time.sleep(1)
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.get("/get_available_cameras")
async def get_available_cameras():
    """Get list of available camera devices"""
    available_cameras = []
    
    # For local development, try to detect physical cameras
    print("Checking for local camera devices...")
    for i in range(5):  # Check first 5 indices
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)  # Use CAP_ANY to be more flexible
            if cap.isOpened():
                # Try to read a frame to confirm it's working
                ret, _ = cap.read()
                if ret:
                    # Camera works and can read frames
                    available_cameras.append({"id": str(i), "name": f"Camera {i}"})
                cap.release()
        except Exception as e:
            print(f"Error checking camera {i}: {e}")
            # Continue to next camera index
    
    # Add browser camera option at the top
    available_cameras.insert(0, {"id": "browser", "name": "Use Browser Camera (recommended for cloud)"})
    
    # If no cameras found at all, provide a message
    if len(available_cameras) <= 1:  # Only has browser camera
        available_cameras.append({"id": "no_cam", "name": "No physical cameras detected"})
    
    print(f"Found {len(available_cameras)} available cameras: {available_cameras}")
    return {"cameras": available_cameras}

def process_webcam_stream(use_tracking: bool = True, obj_id: Optional[int] = None):
    """Process the webcam stream and yield frames with detection/tracking"""
    global track_history, box_size_history, direction_history, speed_history, selected_object_id, robust_tracker
    global webcam_paused, webcam_frame_buffer, just_resumed
    
    # Set selected object ID from parameter if provided
    if obj_id is not None:
        selected_object_id = obj_id
    
    # Initialize tracking
    tracking_enabled = use_tracking
    
    # Force tracking enabled if we have a selected object
    if selected_object_id is not None:
        tracking_enabled = True
        print(f"Webcam tracking forced enabled for object ID: {selected_object_id}")
    
    # For optical flow
    prev_frame = None
    
    # Frame buffer for smoother tracking (keeps the last few frames)
    frame_buffer = []
    BUFFER_SIZE = 3
    
    # Performance monitoring variables
    frame_times = []
    last_frame_time = time.time()
    processing_start_time = time.time()
    
    # Initialize frame counter
    frame_count = 0
    
    # Store current frame for pause functionality
    last_sent_frame = None
    
    # Reset tracking cache only once at stream start
    reset_tracking_cache()
    
    while running and webcam_active:
        # If paused, use the last frame instead of getting a new one
        if webcam_paused:
            if last_sent_frame is not None:
                # Use the last frame we sent
                frame = last_sent_frame.copy()
                # Add a "PAUSED" indicator to the frame
                cv2.putText(frame, "PAUSED", (STD_WIDTH//2 - 80, STD_HEIGHT//2), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                # Yield the frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Slow down when paused to reduce CPU usage
                time.sleep(0.1)
                continue
            else:
                # If no frame is available yet, wait
                time.sleep(0.03)
                continue
        
        # Set flag if we just resumed from pause
        if just_resumed:
            print("Webcam just resumed - maintaining tracking history")
            just_resumed = False  # Reset the flag
            # We don't reset the tracking cache here to maintain trajectory
        
        # Get latest frame from buffer
        frame = get_webcam_frame()
        if frame is None:
            # No frame available yet, wait a moment
            time.sleep(0.03)
            continue
            
        frame_start_time = time.time()
        frame_count += 1
        
        # Resize frame to standard dimensions to ensure consistent processing
        frame = cv2.resize(frame, (STD_WIDTH, STD_HEIGHT), interpolation=cv2.INTER_AREA)
        
        # Add to frame buffer for stabilization
        frame_buffer.append(frame.copy())
        if len(frame_buffer) > BUFFER_SIZE:
            frame_buffer.pop(0)
        
        # Store for optical flow consistency check
        if prev_frame is None:
            prev_frame = frame.copy()
            continue
            
        # Check if frame is similar enough to previous for optical flow
        frame_diff = cv2.absdiff(frame, prev_frame)
        diff_mean = np.mean(frame_diff)
        
        # If frames are too different, reset tracking ONLY if the difference is extreme
        # This makes tracking more persistent and stable for webcam
        if diff_mean > 80 and tracking_enabled:  # Higher threshold for webcam mode
            print(f"Warning: Very large frame difference detected ({diff_mean:.1f}). Resetting tracking.")
            reset_tracking_cache()
        
        # Run detection/tracking on the frame
        try:
            # Run detection or tracking based on settings
            if tracking_enabled:
                # Use the robust tracker if it exists
                if robust_tracker is not None and selected_object_id is not None:
                    # Update with the current frame (custom tracking)
                    success, pos = robust_tracker.update_webcam(frame)
                    if success:
                        # Extract position
                        x, y, w, h = pos
                        
                        # Add to tracking history
                        track_id = 1  # Use a fixed track ID for robust tracker
                        
                        # Update trajectory
                        track_history[track_id].append((float(x), float(y)))
                        box_size_history[track_id].append((float(w), float(h)))
                        
                        # Limit trajectory history 
                        if len(track_history[track_id]) > 90:  # Keep more points for webcam mode
                            track_history[track_id].pop(0)
                            box_size_history[track_id].pop(0)
                        
                        # Draw bounding box and track ID
                        x1, y1 = int(x - w/2), int(y - h/2)
                        x2, y2 = int(x + w/2), int(y + h/2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Draw trajectory line - make it more prominent
                        if len(track_history[track_id]) > 1:
                            points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(frame, [points], False, (0, 255, 255), 2)
                            
                            # Calculate direction if we have enough history
                            if len(track_history[track_id]) >= 5:
                                # Use the last 5 points to calculate direction
                                last_points = track_history[track_id][-5:]
                                if len(last_points) >= 2:
                                    start_x, start_y = last_points[0]
                                    end_x, end_y = last_points[-1]
                                    
                                    # Calculate direction vector
                                    dir_x = end_x - start_x
                                    dir_y = end_y - start_y
                                    
                                    # Calculate direction angle in degrees
                                    angle = math.degrees(math.atan2(dir_y, dir_x))
                                    
                                    # Convert to compass direction
                                    if angle < 0:
                                        angle += 360
                                        
                                    # Map angle to cardinal direction
                                    directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
                                    index = int((angle + 22.5) % 360 / 45)
                                    cardinal = directions[index]
                                    
                                    # Store direction
                                    direction_history[track_id] = cardinal
                                    
                                    # Calculate speed (pixels per frame)
                                    distance = math.sqrt(dir_x**2 + dir_y**2)
                                    frames = len(last_points) - 1
                                    speed_px = distance / frames if frames > 0 else 0
                                    
                                    # Store speed
                                    speed_history[track_id] = speed_px
                                    
                                    # Display direction and speed more prominently
                                    info_text = f"Dir: {cardinal}, Speed: {speed_px:.1f}"
                                    cv2.putText(frame, info_text, (x1, y2 + 20), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                    
                                    # Draw direction arrow
                                    arrow_length = 40
                                    arrow_end_x = int(x + dir_x * arrow_length / distance if distance > 0 else x)
                                    arrow_end_y = int(y + dir_y * arrow_length / distance if distance > 0 else y)
                                    cv2.arrowedLine(frame, (int(x), int(y)), (arrow_end_x, arrow_end_y), 
                                                 (0, 255, 255), 2, tipLength=0.3)
                    
                # Also run YOLOv8 tracking to detect other objects
                results = model.track(frame, persist=True, device=device, half=True)
            else:
                results = model.predict(frame, device=device, half=True)
                
            # Process results for visualization
            # Get bounding boxes, classes and tracking IDs
            if hasattr(results[0], 'boxes'):
                # Process boxes - unified format across detection and tracking
                boxes = results[0].boxes.xywh.cpu().numpy()  # center_x, center_y, width, height
                classes = results[0].boxes.cls.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                # Check if tracking IDs are available
                track_ids = None
                if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy()
                    
                # Draw boxes and tracking information
                for i, box in enumerate(boxes):
                    x, y, w, h = box
                    cls = int(classes[i])
                    conf = confidences[i]
                    
                    # Skip this box if we're already tracking with the robust tracker
                    if robust_tracker is not None and selected_object_id is not None:
                        # If the robust tracker is active, skip YOLOv8 visualization for this object
                        # to avoid double drawing
                        # Simple IOU check to avoid duplicate boxes
                        x1, y1 = int(x - w/2), int(y - h/2)
                        x2, y2 = int(x + w/2), int(y + h/2)
                        rob_x, rob_y, rob_w, rob_h = robust_tracker.get_position()
                        rob_x1, rob_y1 = int(rob_x - rob_w/2), int(rob_y - rob_h/2)
                        rob_x2, rob_y2 = int(rob_x + rob_w/2), int(rob_y + rob_h/2)
                        
                        # Calculate intersection
                        ix1 = max(x1, rob_x1)
                        iy1 = max(y1, rob_y1)
                        ix2 = min(x2, rob_x2)
                        iy2 = min(y2, rob_y2)
                        
                        if ix2 > ix1 and iy2 > iy1:
                            # Boxes overlap, calculate IoU
                            box_area = (x2 - x1) * (y2 - y1)
                            rob_area = (rob_x2 - rob_x1) * (rob_y2 - rob_y1)
                            intersection = (ix2 - ix1) * (iy2 - iy1)
                            union = box_area + rob_area - intersection
                            iou = intersection / union if union > 0 else 0
                            
                            if iou > 0.5:
                                # Skip this box - it's likely the same object
                                continue
                    
                    # Draw bounding box
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    
                    # Calculate color based on class
                    color = (0, 255, 0)  # Default green
                    
                    # If we have a track ID, use it for coloring and tracking
                    if track_ids is not None:
                        track_id = int(track_ids[i])
                        
                        # Generate consistent color for this ID
                        color_r = (track_id * 5) % 255
                        color_g = (track_id * 10) % 255
                        color_b = (track_id * 20) % 255
                        color = (color_b, color_g, color_r)
                        
                        # Draw ID on the box
                        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # If this is the specific object we're tracking, highlight it
                        if selected_object_id is not None and track_id == selected_object_id and robust_tracker is None:
                            # Highlight selected object
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                            
                            # Draw trajectory
                            track_history[track_id].append((float(x), float(y)))
                            box_size_history[track_id].append((float(w), float(h)))
                            
                            # Limit trajectory history
                            if len(track_history[track_id]) > 90:  # Keep more points for webcam mode
                                track_history[track_id].pop(0)
                                box_size_history[track_id].pop(0)
                                
                            # Draw trajectory line
                            points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(frame, [points], False, (0, 255, 255), 2)
                            
                            # Calculate direction if we have enough history
                            if len(track_history[track_id]) >= 5:
                                # Use the last 5 points to calculate direction
                                last_points = track_history[track_id][-5:]
                                if len(last_points) >= 2:
                                    start_x, start_y = last_points[0]
                                    end_x, end_y = last_points[-1]
                                    
                                    # Calculate direction vector
                                    dir_x = end_x - start_x
                                    dir_y = end_y - start_y
                                    
                                    # Calculate direction angle in degrees
                                    angle = math.degrees(math.atan2(dir_y, dir_x))
                                    
                                    # Convert to compass direction
                                    if angle < 0:
                                        angle += 360
                                        
                                    # Map angle to cardinal direction
                                    directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
                                    index = int((angle + 22.5) % 360 / 45)
                                    cardinal = directions[index]
                                    
                                    # Store direction
                                    direction_history[track_id] = cardinal
                                    
                                    # Calculate speed (pixels per frame)
                                    distance = math.sqrt(dir_x**2 + dir_y**2)
                                    frames = len(last_points) - 1
                                    speed_px = distance / frames if frames > 0 else 0
                                    
                                    # Store speed
                                    speed_history[track_id] = speed_px
                                    
                                    # Display direction and speed
                                    info_text = f"Dir: {cardinal}, Speed: {speed_px:.1f}"
                                    cv2.putText(frame, info_text, (x1, y2 + 20), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                    
                                    # Draw direction arrow
                                    arrow_length = 40
                                    arrow_end_x = int(x + dir_x * arrow_length / distance if distance > 0 else x)
                                    arrow_end_y = int(y + dir_y * arrow_length / distance if distance > 0 else y)
                                    cv2.arrowedLine(frame, (int(x), int(y)), (arrow_end_x, arrow_end_y), 
                                                 (0, 255, 255), 2, tipLength=0.3)
                        else:
                            # Regular box for other tracked objects
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    else:
                        # Just draw regular detection boxes
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Display class and confidence
                        label = f"Class:{cls}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Calculate FPS
            frame_time = time.time() - frame_start_time
            frame_times.append(frame_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            avg_frame_time = sum(frame_times) / len(frame_times)
            fps_text = f"FPS: {1/avg_frame_time:.1f}"
            
            # Add processing mode and FPS to the frame
            mode_text = "Tracking" if tracking_enabled else "Detection"
            if selected_object_id is not None:
                mode_text += f" (Tracking ID: {selected_object_id})"
                
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Store the previous frame for optical flow
            prev_frame = frame.copy()
            
            # Store the current frame for pause functionality
            last_sent_frame = frame.copy()
            
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
            # Update frame processing time
            last_frame_time = time.time()
            
        except Exception as e:
            print(f"Error processing webcam frame: {e}")
            import traceback
            traceback.print_exc()
            # Brief pause to prevent excessive error messages
            time.sleep(0.1)

@app.post("/pause_webcam")
async def pause_webcam():
    """Pause webcam streaming without stopping the camera"""
    global webcam_paused
    
    if not webcam_active:
        return {"success": False, "message": "No active webcam to pause"}
    
    webcam_paused = True
    print("Webcam stream paused")
    return {"success": True, "message": "Webcam paused"}

@app.post("/resume_webcam")
async def resume_webcam():
    """Resume webcam streaming after pause"""
    global webcam_paused, just_resumed
    
    if not webcam_active:
        return {"success": False, "message": "No active webcam to resume"}
    
    webcam_paused = False
    just_resumed = True  # Set flag to properly handle tracking after resume
    print("Webcam stream resumed")
    return {"success": True, "message": "Webcam resumed"}

@app.post("/process_browser_frame")
async def process_browser_frame(frame_data: Dict):
    """Process a frame sent from the browser camera"""
    global track_history, box_size_history, direction_history, speed_history, selected_object_id, robust_tracker
    
    try:
        # Extract the base64 image data
        base64_data = frame_data.get("frame", "").split(",")[1]
        tracking_enabled = frame_data.get("tracking", True)
        object_id = frame_data.get("object_id")
        
        # Set selected object ID if provided
        if object_id is not None:
            selected_object_id = object_id
        
        # Decode the base64 data
        img_bytes = base64.b64decode(base64_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"success": False, "error": "Invalid frame data"}
        
        # Resize frame to standard dimensions
        frame = cv2.resize(frame, (STD_WIDTH, STD_HEIGHT), interpolation=cv2.INTER_AREA)
        
        # Process frame with detection/tracking based on settings
        if tracking_enabled:
            # Use the robust tracker if it exists for specific object tracking
            if robust_tracker is not None and selected_object_id is not None:
                success, pos = robust_tracker.update_webcam(frame)
                if success:
                    # Extract position
                    x, y, w, h = pos
                    
                    # Add to tracking history
                    track_id = 1  # Use a fixed track ID for robust tracker
                    
                    # Update trajectory
                    track_history[track_id].append((float(x), float(y)))
                    box_size_history[track_id].append((float(w), float(h)))
                    
                    # Limit trajectory history 
                    if len(track_history[track_id]) > 90:  # Keep more points for webcam
                        track_history[track_id].pop(0)
                        box_size_history[track_id].pop(0)
                    
                    # Draw bounding box and track ID
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Draw trajectory line - make it more prominent
                    if len(track_history[track_id]) > 1:
                        points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], False, (0, 255, 255), 2)
                        
                        # Calculate direction if we have enough history
                        if len(track_history[track_id]) >= 5:
                            # Use the last 5 points to calculate direction
                            last_points = track_history[track_id][-5:]
                            if len(last_points) >= 2:
                                start_x, start_y = last_points[0]
                                end_x, end_y = last_points[-1]
                                
                                # Calculate direction vector
                                dir_x = end_x - start_x
                                dir_y = end_y - start_y
                                
                                # Calculate direction angle in degrees
                                angle = math.degrees(math.atan2(dir_y, dir_x))
                                
                                # Convert to compass direction
                                if angle < 0:
                                    angle += 360
                                    
                                # Map angle to cardinal direction
                                directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
                                index = int((angle + 22.5) % 360 / 45)
                                cardinal = directions[index]
                                
                                # Store direction
                                direction_history[track_id] = cardinal
                                
                                # Calculate speed (pixels per frame)
                                distance = math.sqrt(dir_x**2 + dir_y**2)
                                frames = len(last_points) - 1
                                speed_px = distance / frames if frames > 0 else 0
                                
                                # Store speed
                                speed_history[track_id] = speed_px
                                
                                # Display direction and speed more prominently
                                info_text = f"Dir: {cardinal}, Speed: {speed_px:.1f}"
                                cv2.putText(frame, info_text, (x1, y2 + 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                
                                # Draw direction arrow
                                arrow_length = 40
                                arrow_end_x = int(x + dir_x * arrow_length / distance if distance > 0 else x)
                                arrow_end_y = int(y + dir_y * arrow_length / distance if distance > 0 else y)
                                cv2.arrowedLine(frame, (int(x), int(y)), (arrow_end_x, arrow_end_y), 
                                             (0, 255, 255), 2, tipLength=0.3)
            
            # Run YOLO detection/tracking
            results = model.track(frame, persist=True, device=device, half=True)
        else:
            # Just run detection without tracking
            results = model.predict(frame, device=device, half=True)
        
        # Process and draw bounding boxes 
        if hasattr(results[0], 'boxes'):
            # Process boxes - unified format across detection and tracking
            boxes = results[0].boxes.xywh.cpu().numpy()  # center_x, center_y, width, height
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Check if tracking IDs are available
            track_ids = None
            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy()
                
            # Draw boxes and tracking information
            for i, box in enumerate(boxes):
                x, y, w, h = box
                cls = int(classes[i])
                conf = confidences[i]
                
                # Skip this box if we're already tracking with the robust tracker
                if robust_tracker is not None and selected_object_id is not None:
                    # Simple IOU check to avoid duplicate boxes
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    rob_x, rob_y, rob_w, rob_h = robust_tracker.get_position()
                    rob_x1, rob_y1 = int(rob_x - rob_w/2), int(rob_y - rob_h/2)
                    rob_x2, rob_y2 = int(rob_x + rob_w/2), int(rob_y + rob_h/2)
                    
                    # Calculate intersection
                    ix1 = max(x1, rob_x1)
                    iy1 = max(y1, rob_y1)
                    ix2 = min(x2, rob_x2)
                    iy2 = min(y2, rob_y2)
                    
                    if ix2 > ix1 and iy2 > iy1:
                        # Boxes overlap, calculate IoU
                        box_area = (x2 - x1) * (y2 - y1)
                        rob_area = (rob_x2 - rob_x1) * (rob_y2 - rob_y1)
                        intersection = (ix2 - ix1) * (iy2 - iy1)
                        union = box_area + rob_area - intersection
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > 0.5:
                            # Skip this box - it's likely the same object
                            continue
                
                # Draw bounding box
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                
                # Calculate color based on class
                color = (0, 255, 0)  # Default green
                
                # If we have a track ID, use it for coloring and tracking
                if track_ids is not None:
                    track_id = int(track_ids[i])
                    
                    # Generate consistent color for this ID
                    color_r = (track_id * 5) % 255
                    color_g = (track_id * 10) % 255
                    color_b = (track_id * 20) % 255
                    color = (color_b, color_g, color_r)
                    
                    # Draw ID on the box
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # If this is the specific object we're tracking, highlight it
                    if selected_object_id is not None and track_id == selected_object_id and robust_tracker is None:
                        # Highlight selected object
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                        
                        # Draw trajectory
                        track_history[track_id].append((float(x), float(y)))
                        box_size_history[track_id].append((float(w), float(h)))
                        
                        # Limit trajectory history
                        if len(track_history[track_id]) > 90:  # Keep more points for browser camera
                            track_history[track_id].pop(0)
                            box_size_history[track_id].pop(0)
                            
                        # Draw trajectory line
                        points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], False, (0, 255, 255), 2)
                        
                        # Calculate direction if we have enough history
                        if len(track_history[track_id]) >= 5:
                            # Use the last 5 points to calculate direction
                            last_points = track_history[track_id][-5:]
                            if len(last_points) >= 2:
                                start_x, start_y = last_points[0]
                                end_x, end_y = last_points[-1]
                                
                                # Calculate direction vector
                                dir_x = end_x - start_x
                                dir_y = end_y - start_y
                                
                                # Calculate direction angle in degrees
                                angle = math.degrees(math.atan2(dir_y, dir_x))
                                
                                # Convert to compass direction
                                if angle < 0:
                                    angle += 360
                                    
                                # Map angle to cardinal direction
                                directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
                                index = int((angle + 22.5) % 360 / 45)
                                cardinal = directions[index]
                                
                                # Store direction
                                direction_history[track_id] = cardinal
                                
                                # Calculate speed (pixels per frame)
                                distance = math.sqrt(dir_x**2 + dir_y**2)
                                frames = len(last_points) - 1
                                speed_px = distance / frames if frames > 0 else 0
                                
                                # Store speed
                                speed_history[track_id] = speed_px
                                
                                # Display direction and speed
                                info_text = f"Dir: {cardinal}, Speed: {speed_px:.1f}"
                                cv2.putText(frame, info_text, (x1, y2 + 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                
                                # Draw direction arrow
                                arrow_length = 40
                                arrow_end_x = int(x + dir_x * arrow_length / distance if distance > 0 else x)
                                arrow_end_y = int(y + dir_y * arrow_length / distance if distance > 0 else y)
                                cv2.arrowedLine(frame, (int(x), int(y)), (arrow_end_x, arrow_end_y), 
                                              (0, 255, 255), 2, tipLength=0.3)
                else:
                    # Regular box for other tracked objects
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Add processing mode label
                mode_text = "Tracking" if tracking_enabled else "Detection"
                if selected_object_id is not None:
                    mode_text += f" (Tracking ID: {selected_object_id})"
                    
                cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Browser Camera Mode", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Convert back to base64 for response
                _, buffer = cv2.imencode('.jpg', frame)
                processed_frame = base64.b64encode(buffer).decode('utf-8')
                
                return {
                    "success": True, 
                    "processed_frame": f"data:image/jpeg;base64,{processed_frame}"
                }
    
    except Exception as e:
        print(f"Error processing browser frame: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/browser_camera_frame")
async def browser_camera_frame(file: UploadFile = File(...)):
    """Process a frame sent from the browser camera"""
    global webcam_frame_buffer, webcam_active
    
    try:
        # Read the image data
        contents = await file.read()
        if not contents:
            return {"success": False, "message": "Empty frame received"}
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decode the image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "message": "Could not decode image"}
        
        # Update the webcam frame buffer
        with webcam_lock:
            webcam_frame_buffer = frame.copy()
            
        # Ensure webcam is marked as active
        webcam_active = True
        
        return {"success": True}
    except Exception as e:
        print(f"Error processing browser camera frame: {e}")
        return {"success": False, "message": str(e)}

@app.post("/clear_data")
async def clear_data():
    """Clear all uploaded videos and cached data without shutting down the server"""
    global track_history, box_size_history, direction_history, speed_history, selected_object_id
    global current_video_path, webcam_frame_buffer, process_video
    
    try:
        print("Starting comprehensive data cleanup...")
        
        # Reset tracking states
        print("Resetting tracking data and application state...")
        reset_tracking_data()
        
        # Reset video-related variables
        current_video_path = None
        selected_object_id = None
        
        # Reset frame index for video processing
        if hasattr(process_video, 'current_frame_index'):
            process_video.current_frame_index = 0
        
        if hasattr(process_video, 'current_frame'):
            process_video.current_frame = None
        
        # Reset webcam buffer if active
        if webcam_active:
            with webcam_lock:
                webcam_frame_buffer = None
        
        # Clear PyTorch CUDA cache if using GPU
        if torch.cuda.is_available():
            print("Clearing CUDA cache...")
            torch.cuda.empty_cache()
        
        # Clear uploaded videos
        video_files = list(UPLOAD_DIR.glob("*"))
        initial_count = len(video_files)
        print(f"Clearing uploads: Found {initial_count} files to clean up")
        
        # Delete all files in uploads directory
        deleted_count = 0
        for file in video_files:
            try:
                print(f"Deleting {file}")
                file.unlink(missing_ok=True)  # Python 3.8+ supports missing_ok
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        
        # Clear temporary files
        temp_dir = Path("./temp")
        if temp_dir.exists():
            print("Clearing temporary files...")
            temp_files = list(temp_dir.glob("*"))
            for file in temp_files:
                try:
                    if file.is_file():
                        file.unlink(missing_ok=True)
                    elif file.is_dir():
                        shutil.rmtree(file, ignore_errors=True)
                except Exception as e:
                    print(f"Error deleting temp file {file}: {e}")
        
        # Check for OS-specific temp directories
        if sys.platform == 'win32':
            # Windows TEMP files related to our app
            win_temp = Path(os.environ.get('TEMP', '') or os.environ.get('TMP', ''))
            if win_temp.exists():
                for file in win_temp.glob("yolo_*"):
                    try:
                        if file.is_file():
                            file.unlink(missing_ok=True)
                    except Exception as e:
                        print(f"Error deleting Windows temp file {file}: {e}")
        
        # Verify uploads cleanup
        remaining = list(UPLOAD_DIR.glob("*"))
        if remaining:
            print(f"Warning: {len(remaining)} upload files could not be deleted")
            message = f"Cleared {deleted_count} out of {initial_count} files. {len(remaining)} files could not be deleted."
        else:
            print("All uploaded files cleaned up successfully")
            message = f"Successfully cleared all data ({deleted_count} files removed)."
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("Data cleanup complete")
        return {"success": True, "message": message}
    
    except Exception as e:
        print(f"Error clearing data: {e}")
        return {"success": False, "message": f"Error clearing data: {str(e)}"}

@app.get("/clear_data")
async def clear_data_get():
    """GET method to clear all data (for direct browser access)"""
    # Delegate to the POST method implementation
    return await clear_data()

if __name__ == "__main__":
    # Run the app on all network interfaces on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
