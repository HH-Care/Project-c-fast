import cv2
import numpy as np
from fastapi import FastAPI, Request, BackgroundTasks, File, UploadFile, Form, Query, Depends
from fastapi.responses import StreamingResponse, JSONResponse
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
from typing import Optional
from collections import defaultdict

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

# Standard dimensions for video processing
STD_WIDTH = 640
STD_HEIGHT = 480

# Function to safely load the YOLO model
def load_yolo_model(model_path):
    try:
        # Try to load with default settings first
        model = YOLO(model_path)
        
        # Prepare model for inference without fusion
        dummy_img = np.zeros((STD_HEIGHT, STD_WIDTH, 3), dtype=np.uint8)
        _ = model(dummy_img, verbose=False)  # Run once to initialize
        
        # Check if CUDA is available and use it
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"Model loaded successfully on {device}")
        return model
    
    except Exception as e:
        print(f"Error loading model with default settings: {e}")
        print("Trying alternative loading method...")
        
        # Alternative: Load as PyTorch model directly
        try:
            weights = torch.load(model_path, map_location='cpu')
            model = YOLO(model_path)
            model.model = weights['model'].float()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            print(f"Model loaded with alternative method on {device}")
            return model
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            raise

# Load the YOLO model
try:
    # Set CUDA device settings for optimal performance
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache before loading model
        # Configure PyTorch for better memory management
        torch.backends.cudnn.benchmark = True
        
    model = load_yolo_model("models/best.pt")
except Exception as e:
    print(f"Critical error loading model: {e}")
    print("Using basic YOLO model as fallback")
    model = YOLO("yolov8n.pt")  # Use a standard model as fallback
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

# Store the track history
track_history = defaultdict(list)
box_size_history = defaultdict(list)
direction_history = {}
speed_history = {}

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
    """Reset any cached tracking data between frames to avoid pyramid size errors"""
    global model
    # Force reset of tracking buffer by running inference once on a dummy frame
    dummy_frame = np.zeros((STD_HEIGHT, STD_WIDTH, 3), dtype=np.uint8)
    try:
        model(dummy_frame, verbose=False)
    except:
        pass

def process_video(filename: str, use_tracking: bool = True, obj_id: Optional[int] = None):
    """Process the uploaded video and yield frames with detection/tracking"""
    global track_history, box_size_history, direction_history, speed_history, frame_count, fps, running
    global current_video_path, selected_object_id
    
    # Set selected object ID from parameter if provided
    if obj_id is not None:
        selected_object_id = obj_id
    
    # Reset tracking data for new video
    track_history = defaultdict(list)
    box_size_history = defaultdict(list)
    direction_history = {}
    speed_history = {}
    frame_count = 0
    
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
    
    print(f"Processing video: {filename}")
    print(f"Original dimensions: {original_width}x{original_height}, Standardized: {width}x{height}")
    print(f"FPS: {fps}")
    
    # Maximum history length for tracking
    max_history = 60
    
    # Error counter for tracking failures
    tracking_errors = 0
    max_tracking_errors = 3
    tracking_enabled = use_tracking
    
    # Initialize tracking with dummy frames if needed
    prev_frame = None
    frame_index = 0

    while running:
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
            if tracking_enabled:
                results = model.track(frame, persist=True)
            else:
                # Use detection only (no tracking) if tracking is disabled or too many errors
                results = model(frame)
            
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
                    
                    for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                        # Extract box coordinates
                        x, y, w, h = box
                        x, y, w, h = float(x), float(y), float(w), float(h)
                        
                        # Only process the selected object if selective tracking is enabled
                        if selected_object_id is not None and track_id != selected_object_id:
                            # Skip objects that are not selected
                            continue
                        
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
                        
                        # Display information in overlay
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
        mode_text = "Mode: " + ("Tracking" if tracking_enabled else "Detection only")
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
        
        # Add a slight delay to simulate real-time speed
        time.sleep(1/fps)  # Sleep to match the video's frame rate

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        # Yield a multipart JPEG frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.get("/")
async def index(request: Request):
    # Render the HTML page that displays the video stream
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed(
    filename: str = Query(..., description="Filename for uploaded video"),
    tracking: bool = Query(True, description="Enable object tracking"),
    object_id: Optional[int] = Query(None, description="ID of specific object to track")
):
    """Stream video from uploaded file"""
    return StreamingResponse(
        process_video(filename, use_tracking=tracking, obj_id=object_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/upload_video")
async def upload_video(video: UploadFile = File(...)):
    """Upload a video file for processing"""
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
        
        # Return success response with the filename for client-side use
        return {"success": True, "filename": filename}
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/select_object")
async def select_object(object_id: int = Form(...)):
    """Set the object ID to track selectively"""
    global selected_object_id
    
    selected_object_id = object_id
    return {"success": True, "selected_id": object_id}

@app.post("/clear_selection")
async def clear_selection():
    """Clear the selected object ID"""
    global selected_object_id
    
    selected_object_id = None
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
    
    # Log system information
    print("\nSystem Information:")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA Available: No (Running in CPU mode - performance will be limited)")
    
    print(f"\nPython Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"OpenCV Version: {cv2.__version__}")
    
    print("\nServer running on http://0.0.0.0:8000")
    print("To stop the server:")
    print("1. Press 'q' in the terminal")
    print("2. Or visit http://localhost:8000/shutdown in your browser")
    print("3. Or use the 'Shutdown Server' button in the web interface")
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

if __name__ == "__main__":
    # Run the app on all network interfaces on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
