#!/usr/bin/env python3
"""
Demo script to visualize the bird's eye view transformation
Works without a camera by using a synthetic test image.
"""

import cv2
import numpy as np


def create_test_image(width=640, height=480):
    """Create a synthetic test image simulating a camera view of ground plane."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Sky background (top portion)
    img[:height//2, :] = [200, 180, 150]  # Light blue-gray sky
    
    # Ground with perspective grid
    ground_color = [80, 120, 80]  # Greenish ground
    img[height//2:, :] = ground_color
    
    # Draw perspective grid lines on ground
    vanishing_point = (width // 2, height // 3)
    
    # Vertical lines (converging to vanishing point)
    for x in range(0, width + 1, width // 8):
        cv2.line(img, (x, height), vanishing_point, (60, 100, 60), 2)
    
    # Horizontal lines (getting closer together toward horizon)
    for i, y in enumerate(range(height - 20, height // 2, -30)):
        # Lines get shorter as they go up
        factor = (y - height // 2) / (height // 2)
        x_offset = int((width // 2) * (1 - factor))
        cv2.line(img, (x_offset, y), (width - x_offset, y), (60, 100, 60), 2)
    
    # Add some objects on the ground
    # Cone markers (perspective scaled)
    objects = [
        (width // 4, height - 100, 40),      # Left near
        (3 * width // 4, height - 100, 40),  # Right near
        (width // 3, height - 200, 25),       # Left middle
        (2 * width // 3, height - 200, 25),   # Right middle
        (width // 2 - 50, height - 280, 15),  # Left far
        (width // 2 + 50, height - 280, 15),  # Right far
    ]
    
    for x, y, size in objects:
        # Draw cone
        pts = np.array([
            [x, y],
            [x - size, y + size * 2],
            [x + size, y + size * 2]
        ], np.int32)
        cv2.fillPoly(img, [pts], (0, 100, 255))  # Orange cone
        cv2.polylines(img, [pts], True, (0, 50, 200), 2)
    
    # Add text label
    cv2.putText(img, "Camera View (Perspective)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return img


def apply_birds_eye_transform(img, src_points, dst_points, output_size):
    """Apply perspective transformation."""
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, M, output_size)
    return warped


def add_grid(img, spacing=50):
    """Add reference grid to image."""
    h, w = img.shape[:2]
    overlay = img.copy()
    
    for x in range(0, w, spacing):
        cv2.line(overlay, (x, 0), (x, h), (150, 150, 150), 1)
    for y in range(0, h, spacing):
        cv2.line(overlay, (0, y), (w, y), (150, 150, 150), 1)
    
    return cv2.addWeighted(overlay, 0.7, img, 0.3, 0)


def main():
    """Run the demonstration."""
    print("=" * 50)
    print("Bird's Eye View Transformation Demo")
    print("=" * 50)
    
    # Create test image
    width, height = 640, 480
    test_img = create_test_image(width, height)
    
    # Define source points (trapezoid on ground plane in camera view)
    src_points = np.array([
        [width * 0.1, height * 0.95],    # Bottom-left
        [width * 0.9, height * 0.95],    # Bottom-right
        [width * 0.65, height * 0.55],   # Top-right
        [width * 0.35, height * 0.55],   # Top-left
    ], dtype=np.float32)
    
    # Define destination points (rectangle in output)
    margin = 50
    output_w, output_h = 640, 480
    dst_points = np.array([
        [margin, output_h - margin],           # Bottom-left
        [output_w - margin, output_h - margin], # Bottom-right
        [output_w - margin, margin],           # Top-right
        [margin, margin],                      # Top-left
    ], dtype=np.float32)
    
    # Apply transformation
    bev_img = apply_birds_eye_transform(
        test_img, src_points, dst_points, (output_w, output_h)
    )
    
    # Draw ROI on original image
    display_img = test_img.copy()
    pts = src_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(display_img, [pts], True, (0, 255, 0), 3)
    for i, pt in enumerate(src_points):
        cv2.circle(display_img, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)
        cv2.putText(display_img, str(i+1), (int(pt[0])+15, int(pt[1])-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add grid to bird's eye view
    bev_with_grid = add_grid(bev_img)
    cv2.putText(bev_with_grid, "Bird's Eye View (Top-Down)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Create side-by-side comparison
    comparison = np.hstack([display_img, bev_with_grid])
    
    # Add divider line
    cv2.line(comparison, (width, 0), (width, height), (255, 255, 255), 2)
    
    # Display
    print("\nShowing transformation demo...")
    print("Press any key to close.\n")
    
    cv2.imshow("Bird's Eye View Demo", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the result
    cv2.imwrite("bev_demo_output.png", comparison)
    print("Saved demo output to: bev_demo_output.png")
    
    # Print transformation info
    print("\nTransformation Points:")
    print("Source (Camera View):")
    for i, pt in enumerate(src_points):
        print(f"  Point {i+1}: ({pt[0]:.0f}, {pt[1]:.0f})")
    
    print("\nDestination (Bird's Eye View):")
    for i, pt in enumerate(dst_points):
        print(f"  Point {i+1}: ({pt[0]:.0f}, {pt[1]:.0f})")


if __name__ == "__main__":
    main()
