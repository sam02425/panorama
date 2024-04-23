# USAGE
# python realtime_stitching.py

# import the necessary packages
from __future__ import print_function
from esd.basicmotiondetector import BasicMotionDetector
from esd.panorama_2cam import Stitcher
from imutils.video import VideoStream
import numpy as np
import datetime
import imutils
import time
import cv2

# Initialize the video streams and allow them to warmup
print("[INFO] starting cameras...")
leftStream = VideoStream(src=1).start()
rightStream = VideoStream(src=2).start()
time.sleep(2.0)

# Initialize the image stitcher, motion detector, and total number of frames read
stitcher = Stitcher()
motion = BasicMotionDetector(minArea=500)
total = 0

# Loop over frames from the video streams
while True:
    # Grab the frames from their respective video streams
    left = leftStream.read()
    right = rightStream.read()

    # Capture timestamps
    timestamp_left = time.time()
    timestamp_right = time.time()

    # Print timestamps
    print("Left timestamp:", timestamp_left)
    print("Right timestamp:", timestamp_right)

    # Resize the frames
    left = imutils.resize(left, width=400)
    right = imutils.resize(right, width=400)

    # Stitch the frames together to form the panorama
    result = stitcher.stitch([left, right])

    # No homography could be computed
    if result is None:
        print("[INFO] homography could not be computed")
        break

    # Convert the panorama to grayscale, blur it slightly, update the motion detector
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    locs = motion.update(gray)

    # Only process the panorama for motion if a nice average has been built up
    if total > 32 and len(locs) > 0:
        # Initialize the minimum and maximum (x, y)-coordinates
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

        # Loop over the locations of motion and accumulate the minimum and maximum locations of the bounding boxes
        for l in locs:
            (x, y, w, h) = cv2.boundingRect(l)
            (minX, maxX) = (min(minX, x), max(maxX, x + w))
            (minY, maxY) = (min(minY, y), max(maxY, y + h))

        # Draw the bounding box
        cv2.rectangle(result, (minX, minY), (maxX, maxY), (0, 0, 255), 3)

    # Increment the total number of frames read and draw the timestamp on the image
    total += 1
    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(result, ts, (10, result.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Display the resulting image
    cv2.imshow("Result", result)

    # Display left and right frames side by side
    combined_frame = cv2.hconcat([left, right])
    cv2.imshow("Left and Right Frames", combined_frame)

    # Check for user input
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
leftStream.stop()
rightStream.stop()