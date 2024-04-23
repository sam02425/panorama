import cv2

# Initialize the camera capture objects
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

# Set the camera properties (optional)
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the cameras are opened successfully
if not cap_left.isOpened() or not cap_right.isOpened():
    print("Failed to open cameras")
    exit()

# Create a stitcher object
stitcher = cv2.Stitcher.create()

while True:
    # Capture frames from the cameras
    ret_left, left = cap_left.read()
    ret_right, right = cap_right.read()

    # Check if the frames are captured successfully
    if not ret_left or not ret_right:
        print("Failed to capture frames from one or more cameras")
        break

    # Resize the frames to a consistent size
    left = cv2.resize(left, (400, 400))
    right = cv2.resize(right, (400, 400))

    # Stitch the frames
    status, stitched = stitcher.stitch((left, right))

    # Check if the stitching is successful
    if status == cv2.Stitcher_OK:
        # Display the stitched image
        cv2.imshow("Stitched Image", stitched)
    else:
        print("Failed to stitch the images")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera capture objects and close windows
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
