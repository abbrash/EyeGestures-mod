# EyeGestures Technical Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Dependencies](#dependencies)
5. [API Reference](#api-reference)
6. [Usage Examples](#usage-examples)
7. [Configuration](#configuration)
8. [Performance Considerations](#performance-considerations)
9. [Development Guidelines](#development-guidelines)
10. [Troubleshooting](#troubleshooting)

## Project Overview

EyeGestures is an open-source eye-tracking library that enables gaze-controlled computer interfaces using standard webcams and phone cameras. The project aims to democratize eye-tracking technology by eliminating the need for expensive specialized hardware.

### Key Features
- **Multi-version Support**: Three distinct tracking engines (V1, V2, V3)
- **Real-time Processing**: Optimized for live video streams
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Accessibility Focus**: Designed for assistive technology applications
- **Machine Learning Integration**: Advanced calibration and prediction algorithms

### Mission Statement
The project's mission is to make eye-tracking technology accessible to as many people as possible, particularly for accessibility applications, while providing an alternative control method for all users.

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  Face Detection │───▶│  Eye Tracking   │
│   (Webcam)      │    │   (MediaPipe)   │    │   (Custom ML)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Calibration   │◀───│  Gaze Mapping   │◀───│  Event System   │
│   (ML Models)   │    │  (Screen Coord) │    │  (Fixation/Sacc)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Version Comparison

| Feature | V1 (Legacy) | V2 (Stable) | V3 (Latest) |
|---------|-------------|-------------|-------------|
| **Algorithm** | Model-based | Hybrid (V1 + ML) | Pure ML |
| **Performance** | Moderate | Good | Excellent |
| **Accuracy** | Basic | High | Highest |
| **Calibration** | Manual | Semi-automatic | Automatic |
| **Recommended** | ❌ No | ✅ Yes | ✅ Yes |

## Core Components

### 1. Face Detection (`face.py`)

**Purpose**: Detects and extracts facial landmarks using MediaPipe.

**Key Classes**:
- `FaceFinder`: Initializes MediaPipe face mesh detection
- `Face`: Manages face processing and eye extraction

**MediaPipe Configuration**:
- **Total Landmarks**: 468 facial landmarks (MediaPipe Face Mesh)
- **Detection Confidence**: 0.5 (minimum)
- **Tracking Confidence**: 0.5 (minimum)
- **Refined Landmarks**: Enabled for iris detection
- **Static Image Mode**: Disabled (optimized for video streams)

**Key Methods**:
```python
def find(self, image) -> face_mesh
def process(self, image, face) -> None
def getBoundingBox(self) -> (x, y, width, height)
def _landmarks(self, face) -> np.array  # Converts to pixel coordinates
```

**Landmark Processing**:
- Converts normalized MediaPipe coordinates (0-1) to pixel coordinates
- Extracts full face bounding box for ROI calculations
- Provides offset calculations for eye region extraction

### 2. Eye Processing (`eye.py`)

**Purpose**: Extracts and processes individual eye data from facial landmarks.

**Eye Landmark Specifications**:
- **Left Eye Contour**: Uses MediaPipe's `FACEMESH_LEFT_EYE` landmarks
- **Right Eye Contour**: Uses MediaPipe's `FACEMESH_RIGHT_EYE` landmarks
- **Left Pupil**: Landmark index 473 (MediaPipe iris center)
- **Right Pupil**: Landmark index 468 (MediaPipe iris center)
- **Eye Scale**: (150, 100) pixels for processing

**Key Features**:
- Eye landmark extraction and contour detection
- Blink detection (height ≤ 3 pixels threshold)
- Pupil position tracking with iris landmarks
- Gaze vector calculation using eye geometry
- Eye region extraction with 2-pixel margin

**Blink Detection Algorithm**:
```python
def getBlink(self) -> bool:
    return (self.height) <= 3  # 2x margin for robustness
```

**Gaze Vector Calculation**:
- Uses eye contour landmarks relative to pupil position
- Applies vector mathematics for gaze direction
- Includes correction factors for individual calibration
- Returns normalized gaze vector for screen mapping

**Key Methods**:
```python
def update(self, image, landmarks, offset) -> None
def getBlink(self) -> bool
def getGaze(self, gaze_buffor, y_correction, x_correction) -> np.array
def getLandmarks(self) -> np.array
def getBoundingBox(self) -> (x, y, width, height)
def getOpenness(self) -> float  # Eye openness ratio
```

### 3. Gaze Estimation (`gazeEstimator.py`)

**Purpose**: Core tracking engine that combines eye data to estimate gaze direction.

**Key Classes**:
- `GazeTracker`: Main tracking engine for V1
- `EyeProcessor`: Processes individual eye data

**Key Methods**:
```python
def estimate(self, image, display, context_id, calibration, ...) -> Gevent
def getFeatures(self, image) -> face_mesh
```

### 4. Calibration Systems

#### V1 Calibration (`calibration_v1.py`)
- **Type**: Position-based calibration
- **Method**: Manual point collection at screen edges
- **Use Case**: Legacy support

#### V2 Calibration (`calibration_v2.py`)
- **Type**: Machine learning-based with dual algorithm system
- **Primary Algorithm**: Ridge Regression (α=0.5)
- **Secondary Algorithm**: LassoCV (cross-validation=50, max_iter=10000)
- **Features**: Automatic model switching, asynchronous fitting, thread-safe operations

**Algorithm Progression**:
1. **Initial Phase**: Ridge Regression for fast initial calibration
2. **Refinement Phase**: LassoCV for improved accuracy with feature selection
3. **Automatic Switching**: Based on data availability and convergence

**Calibration Parameters**:
- **Precision Limit**: 50 pixels
- **Precision Step**: 10 pixels
- **Acceptance Radius**: 500 pixels (default)
- **Calibration Radius**: 1000 pixels (default)
- **Minimum Samples**: 30 points per calibration point

**Threading and Performance**:
- **Asynchronous Fitting**: Background model training
- **Thread Safety**: Lock-based synchronization
- **Memory Management**: Automatic cleanup of training threads

**Key Methods**:
```python
def add(self, x, y) -> None                    # Add training sample
def predict(self, x) -> np.array              # Predict gaze point
def movePoint(self) -> None                   # Advance to next calibration point
def whichAlgorithm(self) -> str               # Current algorithm status
def isReadyToMove(self) -> bool               # Check if enough samples collected
def insideClbRadius(self, point, w, h) -> bool # Check calibration radius
def insideAcptcRadius(self, point, w, h) -> bool # Check acceptance radius
```

### 5. Event System (`gevent.py`)

**Purpose**: Defines data structures for gaze events and calibration events.

**Key Classes**:
- `Gevent`: Contains gaze tracking results
- `Cevent`: Contains calibration information

**Gevent Properties**:
```python
class Gevent:
    point: np.array      # Gaze coordinates
    blink: bool          # Blink detection
    fixation: float      # Fixation strength (0-1)
    saccades: bool       # Saccadic movement detection
    sub_frame: np.array  # Processed eye region
```

### 6. Screen Tracking (`screenTracker/`)

**Purpose**: Advanced screen mapping and ROI management.

**Key Components**:
- `ScreenManager`: Main screen processing logic
- `ScreenProcessor`: Coordinate transformations
- `Clusters`: Gaze point clustering using DBSCAN
- `Heatmap`: Visual attention mapping

**Clustering Algorithm (DBSCAN)**:
- **Epsilon**: 12 pixels (cluster radius)
- **Min Samples**: 3 points (minimum cluster size)
- **Purpose**: Identifies main gaze clusters for ROI estimation
- **Output**: Main cluster for screen region of interest

**Heatmap Processing**:
- **Grid Resolution**: 10x10 pixel bins
- **Increment Step**: 10 units per gaze point
- **Threshold**: 4x increment (40 units) for boundary detection
- **Purpose**: Creates 2D histogram of gaze distribution

**ROI Management**:
- **Dynamic Scaling**: ±10% adjustment based on face size changes
- **Edge Detection**: Automatic boundary detection
- **Coordinate Transformation**: Screen ↔ Display coordinate mapping

### 7. Utility Functions (`utils.py`)

**Purpose**: Common utilities and helper functions.

**Key Features**:
- `VideoCapture`: Bufforless video stream wrapper
- `Buffor`: Circular buffer implementation
- `low_pass_filter_fourier`: Signal filtering
- `recoverable`: Error handling decorator

**Buffer Specifications**:
- **Key Points Buffer**: 10 frames (V3), 20 frames (V2)
- **Gaze Buffer**: 200 frames (V1)
- **Pupil Buffer**: 20 frames per eye
- **Eye Buffer**: 20 frames per eye
- **Fixation Buffer**: 100 frames

**Signal Processing**:
- **Fourier Filter**: Cutoff frequency = 200 Hz
- **Column-wise Processing**: Applied to each coordinate dimension
- **Real-time Filtering**: Maintains temporal consistency

**Video Capture Features**:
- **Bufforless Mode**: Default enabled for real-time processing
- **Camera Auto-detection**: Automatic fallback to next available camera
- **Platform-specific**: Windows uses DirectShow backend
- **File Support**: Video files and pickled frame sequences

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-contrib-python` | ≥4.9.0.80 | Computer vision operations |
| `mediapipe` | ≥0.10.8 | Face and landmark detection |
| `numpy` | ≥1.26.4 | Numerical computations |
| `scikit-learn` | ≥1.3.2 | Machine learning algorithms |
| `scipy` | ≥1.12.0 | Scientific computing |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pygame` | ≥2.5.2 | Example applications |
| `matplotlib` | ≥3.8.2 | Visualization |
| `PyAutoGUI` | ≥0.9.54 | Automation features |
| `pynput` | ≥1.7.6 | Input simulation |

### Installation

```bash
# Basic installation
pip install eyeGestures

# With optional dependencies
pip install eyeGestures[full]

# Manual dependency installation (if needed)
pip install mediapipe scikit-learn opencv-contrib-python
```

## API Reference

### Main Classes

#### EyeGestures_v3 (Recommended)

```python
from eyeGestures import EyeGestures_v3

# Initialize
gestures = EyeGestures_v3(calibration_radius=1000)

# Core method
event, cevent = gestures.step(
    frame,           # Input video frame
    calibrate,       # Boolean: enable calibration
    screen_width,    # Display width
    screen_height,   # Display height
    context="main"   # Context identifier
)

# Configuration
gestures.uploadCalibrationMap(points, context="main")
gestures.setFixation(threshold)
gestures.saveModel(context="main")
gestures.loadModel(model_data, context="main")
```

#### EyeGestures_v2 (Stable)

```python
from eyeGestures import EyeGestures_v2

# Initialize
gestures = EyeGestures_v2(calibration_radius=1000)

# Hybrid configuration
gestures.setClassicalImpact(2)  # V1 influence factor
gestures.enableCNCalib()        # Enable hidden calibration

# Same step() method as V3
```

#### EyeGestures_v1 (Legacy)

```python
from eyeGestures import EyeGestures_v1

# Initialize with ROI parameters
gestures = EyeGestures_v1(roi_x=225, roi_y=105, roi_width=80, roi_height=15)

# Different method signature
event, cevent = gestures.estimate(
    frame, display, context, calibration,
    display_width, display_height,
    display_offset_x, display_offset_y,
    fixation_freeze, freeze_radius,
    offset_x, offset_y
)
```

### Video Capture

```python
from eyeGestures.utils import VideoCapture

# Initialize camera
cap = VideoCapture(0)  # Camera index
cap = VideoCapture("video.mp4")  # Video file
cap = VideoCapture("data.pkl")  # Pickled frames

# Read frames
ret, frame = cap.read()

# Cleanup
cap.close()
```

## Usage Examples

### Basic Eye Tracking

```python
import cv2
from eyeGestures import EyeGestures_v3
from eyeGestures.utils import VideoCapture

# Initialize
gestures = EyeGestures_v3()
cap = VideoCapture(0)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    event, cevent = gestures.step(
        frame, 
        calibrate=True,  # Enable calibration
        screen_width=1920, 
        screen_height=1080
    )
    
    if event:
        # Access gaze data
        gaze_x, gaze_y = event.point
        is_blinking = event.blink
        fixation_strength = event.fixation
        is_saccade = event.saccades
        
        print(f"Gaze: ({gaze_x:.1f}, {gaze_y:.1f})")
        print(f"Blink: {is_blinking}, Fixation: {fixation_strength:.2f}")
```

### Custom Calibration

```python
import numpy as np

# Define custom calibration points (normalized 0-1)
calibration_points = np.array([
    [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],  # Top row
    [0.1, 0.5], [0.5, 0.5], [0.9, 0.5],  # Middle row
    [0.1, 0.9], [0.5, 0.9], [0.9, 0.9]   # Bottom row
])

# Upload custom calibration
gestures.uploadCalibrationMap(calibration_points, context="custom")
```

### Multi-Context Tracking

```python
# Different contexts for different applications
gestures.step(frame, True, 1920, 1080, context="game")
gestures.step(frame, True, 1920, 1080, context="browser")
gestures.step(frame, True, 1920, 1080, context="accessibility")

# Each context maintains separate calibration
```

### Model Persistence

```python
# Save trained model
model_data = gestures.saveModel(context="main")

# Load model in new session
gestures.loadModel(model_data, context="main")
```

## Configuration

### Calibration Parameters

```python
# Calibration radius (pixels)
gestures = EyeGestures_v3(calibration_radius=1000)

# Fixation threshold (0-1, higher = more stable)
gestures.setFixation(0.8)

# V2 specific: Classical impact factor
gestures.setClassicalImpact(2)  # 1/N+1 influence of V2
```

### Performance Tuning

```python
# Reduce calibration points for faster setup
calibration_points = np.array([[0.2, 0.2], [0.8, 0.8]])  # Minimal setup

# Adjust fixation threshold for responsiveness
gestures.setFixation(0.5)  # More responsive, less stable
```

### Camera Configuration

```python
# Camera selection
cap = VideoCapture(0)  # Primary camera
cap = VideoCapture(1)  # Secondary camera

# Bufforless mode (default)
cap = VideoCapture(0, bufforless=True)
```

## Performance Considerations

### Optimization Tips

1. **Camera Distance**: Optimal at arm's length (60-80cm)
2. **Lighting**: Ensure good, even lighting on face
3. **Calibration**: More points = better accuracy, but slower setup
4. **Context Management**: Use separate contexts for different applications
5. **Model Persistence**: Save/load models to avoid recalibration

### Performance Metrics

| Version | FPS | Accuracy | Memory Usage | Processing Time |
|---------|-----|----------|--------------|----------------|
| V1 | ~30 | 70% | Low | 0.011-0.015s |
| V2 | ~45 | 85% | Medium | 0.008-0.012s |
| V3 | ~60 | 90% | Medium | 0.005-0.008s |

### Detailed Performance Analysis

**V3 Engine Optimizations**:
- **Head Position Compensation**: Automatic scaling and offset correction
- **Key Points Buffer**: 10-frame temporal smoothing
- **Fourier Filtering**: 200Hz cutoff for noise reduction
- **Velocity-based Saccade Detection**: Dynamic threshold (max_velocity/4)
- **Averaging Window**: 20-point moving average for stability

**V2 Engine Features**:
- **Hybrid Approach**: Combines V1 geometric + V2 ML predictions
- **Classical Impact Factor**: Configurable V1 influence (default: 2)
- **Hidden Calibration**: Optional invisible V1 calibration
- **Edge-based Calibration**: Automatic calibration at screen boundaries

**Memory Usage Breakdown**:
- **Face Landmarks**: 468 × 2 coordinates = 936 floats
- **Eye Landmarks**: ~16 × 2 coordinates per eye = 64 floats
- **Buffer Storage**: 10-200 frames × landmark count
- **ML Models**: Ridge/LassoCV model parameters

### System Requirements

- **CPU**: Multi-core processor recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Camera**: 720p minimum, 1080p recommended
- **Python**: 3.8 or higher

## Development Guidelines

### Code Structure

```
eyeGestures/
├── __init__.py          # Main API exports
├── eye.py              # Eye processing
├── face.py             # Face detection
├── gazeEstimator.py    # V1 tracking engine
├── calibration_v1.py   # V1 calibration
├── calibration_v2.py   # V2/V3 calibration
├── Fixation.py         # Fixation detection
├── gevent.py           # Event definitions
├── processing.py       # Eye processing utilities
├── gazeContexter.py    # Context management
├── utils.py            # Common utilities
└── screenTracker/      # Advanced screen tracking
    ├── screenTracker.py
    ├── clusters.py
    ├── dataPoints.py
    └── heatmap.py
```

### Adding New Features

1. **Backward Compatibility**: Maintain API compatibility
2. **Error Handling**: Use `@recoverable` decorator
3. **Testing**: Add tests for new functionality
4. **Documentation**: Update this documentation

### Contributing

1. Fork the repository
2. Create feature branch
3. Add tests and documentation
4. Submit pull request

## Troubleshooting

### Common Issues

#### 1. Camera Not Detected
```python
# Try different camera indices
for i in range(5):
    try:
        cap = VideoCapture(i)
        break
    except:
        continue
```

**Platform-specific Solutions**:
- **Windows**: Uses DirectShow backend (cv2.CAP_DSHOW)
- **Linux/macOS**: Uses default OpenCV backend
- **Multiple Cameras**: Automatic fallback to next available index

#### 2. Poor Tracking Accuracy
- Ensure good lighting conditions (avoid backlighting)
- Check camera distance (arm's length: 60-80cm)
- Recalibrate with more points (minimum 9, recommended 25)
- Verify face is fully visible (all 468 landmarks detected)
- Check MediaPipe confidence thresholds (min_detection=0.5, min_tracking=0.5)

#### 3. High CPU Usage
- Reduce calibration points (use 9-point grid instead of 25)
- Lower video resolution (720p instead of 1080p)
- Use V1 for lower-end systems
- Disable unnecessary features (saccade detection, heatmaps)
- Monitor buffer sizes (reduce from 200 to 100 frames)

#### 4. Installation Issues
```bash
# Install dependencies separately
pip install mediapipe
pip install scikit-learn
pip install opencv-contrib-python
pip install eyeGestures

# For specific versions
pip install mediapipe==0.10.8
pip install scikit-learn==1.3.2
pip install opencv-contrib-python==4.9.0.80
```

#### 5. Memory Issues
- Use model persistence to avoid recalibration
- Clear contexts when switching applications
- Monitor buffer sizes in long-running applications
- Reduce buffer sizes for memory-constrained systems

#### 6. Calibration Problems
- **Algorithm Not Switching**: Check if enough samples collected (30+ per point)
- **Poor Convergence**: Increase calibration radius or add more points
- **Model Loading Issues**: Verify pickle serialization compatibility
- **Threading Issues**: Check for deadlocks in asynchronous fitting

#### 7. Landmark Detection Issues
- **Face Not Detected**: Check lighting and face visibility
- **Iris Landmarks Missing**: Ensure MediaPipe refine_landmarks=True
- **Inconsistent Landmarks**: Check for camera motion blur
- **Coordinate Conversion**: Verify image dimensions and scaling

### Debug Mode

```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Check algorithm status
algorithm = gestures.whichAlgorithm(context="main")
print(f"Current algorithm: {algorithm}")
```

### Performance Monitoring

```python
import time

start_time = time.time()
event, cevent = gestures.step(frame, calibrate, width, height)
processing_time = time.time() - start_time

print(f"Processing time: {processing_time:.3f}s")
print(f"FPS: {1/processing_time:.1f}")
```

## License and Support

- **License**: GNU General Public License v3 (GPLv3)
- **Support**: contact@eyegestures.com
- **Community**: [Discord](https://discord.gg/FvagCX8T4h)
- **Documentation**: [GitHub](https://github.com/NativeSensors/EyeGestures)

## Technical Specifications

### MediaPipe Integration Details

**Face Mesh Landmarks (468 total)**:
- **Face Contour**: 17 landmarks (jawline)
- **Left Eyebrow**: 5 landmarks
- **Right Eyebrow**: 5 landmarks
- **Nose**: 4 landmarks (bridge and tip)
- **Left Eye**: 6 landmarks (contour)
- **Right Eye**: 6 landmarks (contour)
- **Outer Lips**: 12 landmarks
- **Inner Lips**: 8 landmarks
- **Iris Landmarks**: 5 per eye (when refine_landmarks=True)

**Eye-Specific Landmarks Used**:
- **Left Eye Contour**: MediaPipe indices from `FACEMESH_LEFT_EYE`
- **Right Eye Contour**: MediaPipe indices from `FACEMESH_RIGHT_EYE`
- **Left Iris Center**: Index 473
- **Right Iris Center**: Index 468

### Data Flow Architecture

```
Video Frame → MediaPipe → 468 Landmarks → Eye Extraction → Gaze Estimation
     ↓              ↓            ↓              ↓              ↓
  BGR→RGB    Face Detection   Coordinate    Blink Detection   Screen Mapping
  Flip H     Confidence>0.5   Conversion    Height≤3px       ML Prediction
```

### Feature Vector Composition

**V3 Feature Vector**:
- Left eye landmarks: ~16 coordinates
- Right eye landmarks: ~16 coordinates  
- Head scale factors: 2 values (x, y)
- Head offset: 2 values (x, y)
- **Total**: ~36 features per frame

**V2 Feature Vector**:
- V1 gaze prediction: 2 coordinates
- Left eye landmarks: ~16 coordinates
- Right eye landmarks: ~16 coordinates
- Eye events: 2 values (blink, fixation)
- **Total**: ~36 features per frame

### Calibration Matrix Default Points

**25-Point Grid** (5×5):
```python
points = [
    [1.0, 0.0], [0.75, 0.0], [0.5, 0.0], [0.25, 0.0], [0.0, 0.0],
    [1.0, 0.25], [0.75, 0.25], [0.5, 0.25], [0.25, 0.25], [0.0, 0.25],
    [1.0, 0.5], [0.75, 0.5], [0.5, 0.5], [0.25, 0.5], [0.0, 0.5],
    [1.0, 0.75], [0.75, 0.75], [0.5, 0.75], [0.25, 0.75], [0.0, 0.75],
    [1.0, 1.0], [0.75, 1.0], [0.5, 1.0], [0.25, 1.0], [0.0, 1.0]
]
```

### Error Handling and Recovery

**Recoverable Decorator**:
```python
@recoverable(ret_error_params=(None, None))
def step(self, frame, calibration, width, height, context="main"):
    # Returns (None, None) on any exception
```

**Common Error Scenarios**:
- MediaPipe face detection failure
- Insufficient calibration data
- Thread synchronization issues
- Memory allocation failures
- Camera stream interruptions

## Version History

- **v3.2.4**: Current version with V3 engine
- **v3.0.0**: Introduction of V3 engine
- **v2.x**: Stable V2 implementation
- **v1.x**: Legacy model-based approach

---

*This documentation is maintained by the EyeGestures development team. For updates and contributions, please visit the project repository.*
