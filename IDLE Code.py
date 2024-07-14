import cv2
import dlib
from scipy.spatial import distance
import time
import numpy as np
import winsound

# Load face detector and predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Constants for eye landmarks
LEFT_EYE_START = 36
LEFT_EYE_END = 41
RIGHT_EYE_START = 42
RIGHT_EYE_END = 47

# Threshold for eye closure detection (adjust as needed)
eye_closure_threshold = 0.2  # Adjust this threshold for your specific case

# Initialize timer variables
start_time_closed = None
eyes_closed_duration = 0
alert_displayed = False

# Function to play a beep sound
def play_beep():
    winsound.Beep(1000, 200)  # You can adjust the frequency (1000 Hz) and duration (200 ms) as needed

def eye_aspect_ratio(eye):
    # Compute the EAR
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)

        # Extract left and right eye coordinates
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(LEFT_EYE_START, LEFT_EYE_END + 1)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(RIGHT_EYE_START, RIGHT_EYE_END + 1)]

        # Calculate EAR for both eyes
        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)

        # Average EAR for both eyes
        ear_avg = (ear_left + ear_right) / 2.0

        # Check if the eyes are closed based on the threshold
        if ear_avg < eye_closure_threshold:
            # Eyes closed or partially closed
            if start_time_closed is None:
                start_time_closed = time.time()
            else:
                eyes_closed_duration = time.time() - start_time_closed
                if eyes_closed_duration >= 1.5 and not alert_displayed:
                    cv2.putText(frame, "Alert: Eyes Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    alert_displayed = True
                    if alert_displayed == True:
                        for _ in range(2):
                            play_beep()# Play beep sound
                elif not alert_displayed:
                    cv2.putText(frame, "Eyes Partially Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Eyes open
            start_time_closed = None
            eyes_closed_duration = 0
            alert_displayed = False
            cv2.putText(frame, "Eyes Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw bounding box around the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw rectangles around the eyes
        cv2.polylines(frame, [np.array(left_eye, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [np.array(right_eye, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
