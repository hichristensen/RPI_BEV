#!/usr/bin/env python3
"""
Bird's Eye View Projection for Raspberry Pi Camera

This script captures images from a Raspberry Pi camera and projects them
onto a ground plane to create a top-down (bird's eye) view. Useful for
robotics, autonomous navigation, and surveillance applications.

Requirements:
    pip install opencv-python numpy picamera2

Usage:
    1. Run in calibration mode first to set up the transformation:
       python birds_eye_view.py --calibrate

    2. Then run normally:
       python birds_eye_view.py
"""

import json
import argparse
from pathlib import Path
import numpy as np
import cv2

# Try to import picamera2 for Raspberry Pi, fall back to regular webcam
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("picamera2 not available, using standard webcam")


class BirdsEyeView:
    """
    Transforms camera images to a bird's eye (top-down) view by applying
    a perspective transformation (homography) to project the image onto
    an assumed ground plane.
    """

    def __init__(self, config_path="bev_config.json"):
        self.config_path = Path(config_path)
        self.camera = None
        self.frame_width = 640
        self.frame_height = 480

        # Source points (corners in camera view) - will be calibrated
        self.src_points = None
        # Destination points (corners in bird's eye view)
        self.dst_points = None
        # Transformation matrix
        self.transform_matrix = None
        self.inverse_matrix = None

        # Output dimensions for bird's eye view
        self.output_width = 640
        self.output_height = 480

        # Load existing calibration if available
        self.load_config()

    def init_camera(self):
        """Initialize the camera (Raspberry Pi camera or webcam)."""
        if PICAMERA_AVAILABLE:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (self.frame_width, self.frame_height), "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            print("Raspberry Pi camera initialized")
        else:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            print("Webcam initialized")

    def capture_frame(self):
        """Capture a single frame from the camera."""
        if PICAMERA_AVAILABLE:
            frame = self.camera.capture_array()
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.camera.read()
            if not ret:
                return None
        return frame

    def release_camera(self):
        """Release camera resources."""
        if self.camera is not None:
            if PICAMERA_AVAILABLE:
                self.camera.stop()
            else:
                self.camera.release()

    def load_config(self):
        """Load calibration configuration from file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.src_points = np.array(config['src_points'], dtype=np.float32)
                self.dst_points = np.array(config['dst_points'], dtype=np.float32)
                self.output_width = config.get('output_width', 640)
                self.output_height = config.get('output_height', 480)
                self.compute_transform()
                print(f"Loaded calibration from {self.config_path}")
                return True
        return False

    def save_config(self):
        """Save calibration configuration to file."""
        config = {
            'src_points': self.src_points.tolist(),
            'dst_points': self.dst_points.tolist(),
            'output_width': self.output_width,
            'output_height': self.output_height
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved calibration to {self.config_path}")

    def compute_transform(self):
        """Compute the perspective transformation matrix."""
        if self.src_points is not None and self.dst_points is not None:
            self.transform_matrix = cv2.getPerspectiveTransform(
                self.src_points, self.dst_points
            )
            self.inverse_matrix = cv2.getPerspectiveTransform(
                self.dst_points, self.src_points
            )

    def set_default_points(self):
        """Set default source and destination points for a typical setup."""
        # Default BEV-RPI
        # source points: trapezoid in lower portion of image
        # These represent the visible ground plane in camera view
        self.src_points = np.array([
            [self.frame_width * 0.1, self.frame_height * 0.9],   # Bottom-left
            [self.frame_width * 0.9, self.frame_height * 0.9],   # Bottom-right
            [self.frame_width * 0.7, self.frame_height * 0.5],   # Top-right
            [self.frame_width * 0.3, self.frame_height * 0.5],   # Top-left
        ], dtype=np.float32)

        # Destination points: rectangle in output image
        margin = 150
        self.dst_points = np.array([
            [margin, self.output_height - margin],               # Bottom-left
            [self.output_width - margin, self.output_height - margin],  # Bottom-right
            [self.output_width - margin, margin],                # Top-right
            [margin, margin],                                    # Top-left
        ], dtype=np.float32)

        self.compute_transform()

    def transform_to_bev(self, frame):
        """Transform a camera frame to bird's eye view."""
        if self.transform_matrix is None:
            print("Error: No transformation matrix. Run calibration first.")
            return frame

        bev = cv2.warpPerspective(
            frame, 
            self.transform_matrix, 
            (self.output_width, self.output_height),
            flags=cv2.INTER_LINEAR
        )
        return bev

    def transform_point_to_bev(self, point):
        """Transform a single point from camera view to bird's eye view."""
        if self.transform_matrix is None:
            return point

        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.transform_matrix)
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))

    def transform_point_from_bev(self, point):
        """Transform a point from bird's eye view back to camera view."""
        if self.inverse_matrix is None:
            return point

        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.inverse_matrix)
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))

    def draw_roi_on_frame(self, frame):
        """Draw the region of interest (source points) on the camera frame."""
        if self.src_points is not None:
            pts = self.src_points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            for i, pt in enumerate(self.src_points):
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
                cv2.putText(frame, str(i+1), (int(pt[0])+10, int(pt[1])-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def add_grid_overlay(self, bev_frame, grid_size=50):
        """Add a grid overlay to the bird's eye view for reference."""
        overlay = bev_frame.copy()

        # Draw vertical lines
        for x in range(0, self.output_width, grid_size):
            cv2.line(overlay, (x, 0), (x, self.output_height), (100, 100, 100), 1)

        # Draw horizontal lines
        for y in range(0, self.output_height, grid_size):
            cv2.line(overlay, (0, y), (self.output_width, y), (100, 100, 100), 1)

        # Blend with original
        return cv2.addWeighted(overlay, 0.7, bev_frame, 0.3, 0)

    def calibrate_interactive(self):
        """Interactive calibration mode to set the transformation points."""
        print("\n=== Bird's Eye View Calibration ===")
        print("Instructions:")
        print("  - Click 4 points on the ground plane in this order:")
        print("    1. Bottom-left corner")
        print("    2. Bottom-right corner")
        print("    3. Top-right corner")
        print("    4. Top-left corner")
        print("  - Press 'r' to reset points")
        print("  - Press 's' to save and exit")
        print("  - Press 'd' to use default points")
        print("  - Press 'q' to quit without saving")
        print("=" * 40)

        self.init_camera()

        points = []
        current_frame = None

        def mouse_callback(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append([x, y])
                print(f"Point {len(points)}: ({x}, {y})")

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)

        while True:
            frame = self.capture_frame()
            if frame is None:
                continue

            current_frame = frame.copy()
            display = frame.copy()

            # Draw existing points
            for i, pt in enumerate(points):
                cv2.circle(display, (pt[0], pt[1]), 8, (0, 0, 255), -1)
                cv2.putText(display, str(i+1), (pt[0]+10, pt[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw polygon if we have enough points
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    cv2.line(display, tuple(points[i]), tuple(points[i+1]), (0, 255, 0), 2)
                if len(points) == 4:
                    cv2.line(display, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)

            # Show preview of bird's eye view if 4 points selected
            if len(points) == 4:
                self.src_points = np.array(points, dtype=np.float32)
                margin = 50
                self.dst_points = np.array([
                    [margin, self.output_height - margin],
                    [self.output_width - margin, self.output_height - margin],
                    [self.output_width - margin, margin],
                    [margin, margin],
                ], dtype=np.float32)
                self.compute_transform()

                bev = self.transform_to_bev(current_frame)
                bev = self.add_grid_overlay(bev)
                cv2.imshow("Bird's Eye Preview", bev)

            # Add instructions to display
            cv2.putText(display, f"Points: {len(points)}/4", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display, "r:reset s:save d:default q:quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow("Calibration", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                points = []
                print("Points reset")
                cv2.destroyWindow("Bird's Eye Preview")

            elif key == ord('d'):
                self.set_default_points()
                points = self.src_points.tolist()
                print("Using default points")

            elif key == ord('s'):
                if len(points) == 4:
                    self.save_config()
                    print("Calibration saved!")
                    break
                else:
                    print("Need 4 points to save")

            elif key == ord('q'):
                print("Calibration cancelled")
                break

        cv2.destroyAllWindows()
        self.release_camera()

    def run(self, show_original=True, show_grid=True):
        """Run the bird's eye view transformation in real-time."""
        if self.transform_matrix is None:
            print("No calibration found. Running calibration...")
            self.calibrate_interactive()
            if self.transform_matrix is None:
                print("Calibration required. Exiting.")
                return

        self.init_camera()

        print("\n=== Bird's Eye View Running ===")
        print("Press 'g' to toggle grid")
        print("Press 'o' to toggle original view")
        print("Press 'c' to recalibrate")
        print("Press 'q' to quit")
        print("=" * 35)

        while True:
            frame = self.capture_frame()
            if frame is None:
                continue

            # Transform to bird's eye view
            bev = self.transform_to_bev(frame)

            if show_grid:
                bev = self.add_grid_overlay(bev)

            # Display bird's eye view
            cv2.imshow("Bird's Eye View", bev)

            # Optionally show original with ROI overlay
            if show_original:
                original = self.draw_roi_on_frame(frame.copy())
                cv2.imshow("Original View", original)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('g'):
                show_grid = not show_grid
            elif key == ord('o'):
                show_original = not show_original
                if not show_original:
                    cv2.destroyWindow("Original View")
            elif key == ord('c'):
                cv2.destroyAllWindows()
                self.release_camera()
                self.calibrate_interactive()
                if self.transform_matrix is not None:
                    self.init_camera()
                else:
                    break

        cv2.destroyAllWindows()
        self.release_camera()


class BirdsEyeViewAdvanced(BirdsEyeView):
    """
    Advanced bird's eye view with camera calibration support.
    Uses camera intrinsic parameters for more accurate transformation.
    """

    def __init__(self, config_path="bev_config.json"):
        super().__init__(config_path)

        # Camera parameters (can be obtained from camera calibration)
        self.camera_matrix = None
        self.dist_coeffs = None

        # Physical setup parameters
        self.camera_height = 0.3  # meters above ground
        self.camera_pitch = 45.0  # degrees, looking down

    def set_camera_parameters(self, fx, fy, cx, cy, dist_coeffs=None):
        """
        Set camera intrinsic parameters.

        Args:
            fx, fy: Focal lengths in pixels
            cx, cy: Principal point (optical center)
            dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]
        """
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        if dist_coeffs is not None:
            self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
        else:
            self.dist_coeffs = np.zeros(5, dtype=np.float32)

    def undistort_frame(self, frame):
        """Remove lens distortion from the frame."""
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        return frame

    def compute_transform_from_pose(self, camera_height, pitch_degrees, 
                                    roll_degrees=0, yaw_degrees=0):
        """
        Compute the bird's eye view transformation from camera pose.

        Args:
            camera_height: Height of camera above ground (meters)
            pitch_degrees: Camera pitch angle (positive = looking down)
            roll_degrees: Camera roll angle
            yaw_degrees: Camera yaw angle
        """
        # Convert to radians
        pitch = np.radians(pitch_degrees)
        roll = np.radians(roll_degrees)
        yaw = np.radians(yaw_degrees)

        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])

        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])

        # Combined rotation
        R = Rz @ Ry @ Rx

        # Translation vector (camera position)
        t = np.array([[0], [0], [camera_height]])

        # If we have camera intrinsics, compute proper homography
        if self.camera_matrix is not None:
            # Homography for ground plane (z=0)
            # H = K * [r1, r2, t] where r1, r2 are first two columns of R
            H = self.camera_matrix @ np.column_stack([R[:, 0], R[:, 1], t.flatten()])

            # Scale and center the output
            scale = self.output_width / (2 * camera_height * np.tan(np.radians(30)))

            # Create output transformation
            T_out = np.array([
                [scale, 0, self.output_width / 2],
                [0, scale, self.output_height / 2],
                [0, 0, 1]
            ])

            self.transform_matrix = T_out @ np.linalg.inv(H)
            self.inverse_matrix = np.linalg.inv(self.transform_matrix)
        else:
            print("Warning: Camera matrix not set. Using default transformation.")
            self.set_default_points()

    def transform_to_bev(self, frame):
        """Transform with optional undistortion."""
        # First undistort if we have calibration
        frame = self.undistort_frame(frame)
        return super().transform_to_bev(frame)


def create_stitched_bev(frames, transforms, output_size):
    """
    Create a stitched bird's eye view from multiple cameras.

    Args:
        frames: List of camera frames
        transforms: List of transformation matrices
        output_size: (width, height) of output image

    Returns:
        Stitched bird's eye view image
    """
    output = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    for frame, transform in zip(frames, transforms):
        warped = cv2.warpPerspective(frame, transform, output_size)

        # Simple blending - take non-zero pixels
        mask = np.any(warped > 0, axis=2)
        output[mask] = warped[mask]

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Bird's Eye View Projection for Raspberry Pi Camera"
    )
    parser.add_argument(
        '--calibrate', '-c',
        action='store_true',
        help='Run in calibration mode'
    )
    parser.add_argument(
        '--config', '-f',
        type=str,
        default='bev_config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--advanced', '-a',
        action='store_true',
        help='Use advanced mode with camera calibration'
    )
    parser.add_argument(
        '--no-grid',
        action='store_true',
        help='Disable grid overlay'
    )
    parser.add_argument(
        '--no-original',
        action='store_true',
        help='Hide original camera view'
    )

    args = parser.parse_args()

    if args.advanced:
        bev = BirdsEyeViewAdvanced(config_path=args.config)
        # Set approximate parameters for Raspberry Pi Camera Module v2
        # (you should calibrate these properly for best results)
        bev.set_camera_parameters(
            fx=1000, fy=1000,  # Approximate focal length
            cx=320, cy=240,    # Principal point (center of image)
            dist_coeffs=None
        )
    else:
        bev = BirdsEyeView(config_path=args.config)

    if args.calibrate:
        bev.calibrate_interactive()
    else:
        bev.run(
            show_original=not args.no_original,
            show_grid=not args.no_grid
        )
        
if __name__ == "__main__":
    main()
