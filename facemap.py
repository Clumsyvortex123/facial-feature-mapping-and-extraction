import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Indices of the landmarks for the eyes, nose, and mouth
landmarks_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163,362, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 464, 398,61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91,61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,336,296,334,293,300,
285,295,282,283,276,107,66,105,63,70,55,65,52,53,46,193,122,196,3,51,45,44,1,19,274,275,281,248,419,351,417,209,49,292,64,235,429,279,358,294,455]

# Initialize Face Mesh
with mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2) as face_mesh:

    while cap.isOpened():
        # Read a frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Draw only selected face landmarks
                for idx in landmarks_indices:
                    landmark = landmarks.landmark[idx]
                    h, w, c = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow('Golden Ratio Face Map', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
