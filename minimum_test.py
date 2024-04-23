import cv2

# Initialize the camera capture object
cap = cv2.VideoCapture(2)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()

while True:
    # Capture frame from the camera
    ret, frame = cap.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Failed to capture frame")
        break

    # Display the captured frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera capture object and close windows
cap.release()
cv2.destroyAllWindows()
