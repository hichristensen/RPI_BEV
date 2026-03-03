# Bird's Eye View Projection for Raspberry Pi Camera

Transform your Raspberry Pi camera view into a top-down (bird's eye) perspective. This is useful for robotics, autonomous vehicles, parking assistance, and surveillance applications.

## How It Works

The system uses **perspective transformation (homography)** to project the camera image onto an assumed ground plane:

```
Camera View                    Bird's Eye View
┌─────────────────┐           ┌─────────────────┐
│                 │           │                 │
│    ╱─────╲      │           │  ┌───────────┐  │
│   ╱       ╲     │    →      │  │           │  │
│  ╱         ╲    │           │  │           │  │
│ ╱___________╲   │           │  └───────────┘  │
└─────────────────┘           └─────────────────┘
   Trapezoid ROI                 Rectangle
```

## Installation

### On Raspberry Pi

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-opencv python3-numpy

# For Raspberry Pi Camera Module
sudo apt install -y python3-picamera2

# Or install via pip
pip install opencv-python numpy picamera2
```

### On Desktop (for testing)

```bash
pip install opencv-python numpy
```

## Quick Start

### 1. Calibrate the System

Run calibration mode to define the ground plane region:

```bash
python birds_eye_view.py --calibrate
```

**Calibration Steps:**
1. Point your camera at the ground
2. Click 4 points defining a rectangular area on the ground (in order: bottom-left, bottom-right, top-right, top-left)
3. Preview the transformation
4. Press 's' to save

### 2. Run the Transformation

```bash
python birds_eye_view.py
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--calibrate, -c` | Run interactive calibration mode |
| `--config FILE, -f FILE` | Specify configuration file path |
| `--advanced, -a` | Use advanced mode with lens distortion correction |
| `--no-grid` | Disable the grid overlay |
| `--no-original` | Hide the original camera view |

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `g` | Toggle grid overlay |
| `o` | Toggle original view |
| `c` | Recalibrate |
| `r` | Reset points (calibration mode) |
| `s` | Save calibration |
| `d` | Use default points |

## Camera Mounting

For best results:

```
        Camera
           │ ↘ (angled down 30-60°)
           │
           │
    ═══════╧═══════  Ground plane
```

- Mount camera 20-100cm above ground
- Angle camera 30-60° downward
- Ensure good lighting on ground area
- Use a fixed mount for stability

## Configuration File

The calibration is saved to `bev_config.json`:

```json
{
  "src_points": [[64, 432], [576, 432], [448, 240], [192, 240]],
  "dst_points": [[50, 430], [590, 430], [590, 50], [50, 50]],
  "output_width": 640,
  "output_height": 480
}
```

## Using in Your Own Code

```python
from birds_eye_view import BirdsEyeView

# Initialize
bev = BirdsEyeView(config_path="bev_config.json")

# Transform a single frame
import cv2
frame = cv2.imread("camera_image.jpg")
top_down = bev.transform_to_bev(frame)

# Transform a point coordinate
camera_point = (320, 400)
ground_point = bev.transform_point_to_bev(camera_point)
```

## Advanced: Camera Calibration

For higher accuracy, use the `BirdsEyeViewAdvanced` class with proper camera calibration:

```python
from birds_eye_view import BirdsEyeViewAdvanced

bev = BirdsEyeViewAdvanced()

# Set camera intrinsic parameters (from calibration)
bev.set_camera_parameters(
    fx=1000,  # Focal length X
    fy=1000,  # Focal length Y  
    cx=320,   # Principal point X
    cy=240,   # Principal point Y
    dist_coeffs=[0.1, -0.2, 0, 0, 0]  # Distortion coefficients
)

# Or compute transform from camera pose
bev.compute_transform_from_pose(
    camera_height=0.5,    # 50cm above ground
    pitch_degrees=45,     # Looking 45° down
    roll_degrees=0,
    yaw_degrees=0
)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not found | Check `libcamera-hello` works, or use USB webcam |
| Distorted output | Recalibrate with more care on point placement |
| Poor transformation | Ensure 4 calibration points form a rectangle on ground |
| Slow performance | Reduce resolution in code |

## Applications

- **Robot navigation**: Top-down view for path planning
- **Parking assistance**: Bird's eye parking camera
- **Object tracking**: Easier to track objects on ground plane
- **Multi-camera stitching**: Combine multiple views into one

## License

MIT License - Free for personal and commercial use.
