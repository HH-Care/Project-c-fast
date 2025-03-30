# Real-time Object Tracking System

A GPU-accelerated object tracking system that provides real-time analysis of video feeds, including:
- Object detection and tracking
- Direction and speed analysis
- 3D movement prediction
- Selective object tracking

## Requirements

- Python 3.8+
- CUDA-compatible GPU
- Required Python packages (listed in requirements.txt)

## AWS EC2 GPU Setup

### 1. Launch an EC2 Instance

1. Choose a GPU-enabled instance type (g4dn.xlarge or similar)
2. Select an AMI with CUDA support (AWS Deep Learning AMI recommended)
3. Configure security group to allow inbound traffic on port 8000

### 2. Connect to your instance

```bash
ssh -i your-key.pem ec2-user@your-instance-ip
```

### 3. Install CUDA (if not already included in the AMI)

Follow NVIDIA's instructions for your specific Ubuntu/Amazon Linux version:
https://developer.nvidia.com/cuda-downloads

Verify installation:
```bash
nvidia-smi
```

### 4. Clone and setup the application

```bash
# Clone the repository
git clone https://your-repository-url.git
cd gpu-realtime

# Make the start script executable
chmod +x start.sh

# Run the application
./start.sh
```

This will:
- Create a Python virtual environment
- Install dependencies
- Create necessary directories
- Download the YOLO model if needed
- Start the application server

### 5. Access the application

Open your browser and navigate to:
```
http://your-instance-ip:8000
```

## Usage

1. Upload a video for analysis
2. Toggle tracking on/off if needed
3. Observe object IDs displayed on detected objects
4. For selective tracking, enter an object ID in the input field
5. Click "Track This Object" to focus on a specific object

## Notes for AWS EC2 Deployment

- **Persistent Storage**: Consider attaching an EBS volume for storing models and videos
- **Security**: Update security groups to restrict access as needed
- **Cost Management**: GPU instances can be expensive. Stop or hibernate when not in use
- **Monitoring**: Use CloudWatch to monitor GPU utilization

## Troubleshooting

- If the model fails to load, check CUDA compatibility
- For memory issues, try reducing STD_WIDTH and STD_HEIGHT in main.py
- If tracking causes errors, try the detection-only mode by toggling tracking off

## Shutdown

Use the "Shutdown Server" button in the web interface or press `q` in the terminal to properly stop the server and clean up temporary files. 