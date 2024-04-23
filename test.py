import cv2
from esd.panorama import Stitcher  # the Stitcher class defined in panorama.py
import numpy as np


# Initialize the camera capture objects
cap_left = cv2.VideoCapture(0)  # Assuming the left camera is connected to index 0
cap_middle = cv2.VideoCapture(1)  # Assuming the middle camera is connected to index 1
cap_right = cv2.VideoCapture(2)  # Assuming the right camera is connected to index 2

# Check if the cameras are opened successfully
if not cap_left.isOpened() or not cap_middle.isOpened() or not cap_right.isOpened():
    print("Failed to open cameras")
    exit()

# Capture frames from the cameras
left = cap_left.read()[1]
middle = cap_middle.read()[1]
right = cap_right.read()[1]

# Check the shape of the images
print("Left image shape:", left.shape)
print("Middle image shape:", middle.shape)
print("Right image shape:", right.shape)

# Resize the images
left = cv2.resize(left, (400, 400))
middle = cv2.resize(middle, (400, 400))
right = cv2.resize(right, (400, 400))

# Check the shape of the resized images
print("Resized left image shape:", left.shape, "dtype:", left.dtype, "min:", np.min(left), "max:", np.max(left))
print("Resized middle image shape:", middle.shape, "dtype:", middle.dtype, "min:", np.min(middle), "max:", np.max(middle))
print("Resized right image shape:", right.shape, "dtype:", right.dtype, "min:", np.min(right), "max:", np.max(right))

if left is None or middle is None or right is None:
    print("One or more resized images are None")
    exit()

if np.any(np.isnan(left)) or np.any(np.isnan(middle)) or np.any(np.isnan(right)):
    print("One or more resized images contain NaN values")
    exit()

if np.any(np.isinf(left)) or np.any(np.isinf(middle)) or np.any(np.isinf(right)):
    print("One or more resized images contain infinite values")
    exit()

# Convert the images to BGR format if needed
if left is not None:
    left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
if middle is not None:
    middle = cv2.cvtColor(middle, cv2.COLOR_RGB2BGR)
if right is not None:
    right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)

# Create an instance of the Stitcher class
stitcher = Stitcher()

# Stitch the images
result = stitcher.stitch([left, middle, right])

# Release the camera capture objects
cap_left.release()
cap_middle.release()
cap_right.release()