import cv2
import tkinter as tk
import mediapipe as mp
import math
import os
from PIL import Image, ImageTk

class FacialPointsApp:
    def __init__(self, root, title):
        self.root = root
        self.root.title(title)
        self.cap = cv2.VideoCapture(0)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.canvas = tk.Canvas(root, width=self.cap.get(3), height=self.cap.get(4))
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        self.captured_image_label = tk.Label(root)
        self.captured_image_label.grid(row=0, column=1, padx=10, pady=10)

        self.btn_capture = tk.Button(root, text="Capture", command=self.capture_image)
        self.btn_capture.grid(row=1, column=0, columnspan=2, pady=10)

        self.update()

        self.save_path = r'D:\face map script\facepics\img1.jpg'  # Update this path to your desired location

    def capture_image(self):
        ret, frame = self.cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        landmarks = self.detect_landmarks(frame_rgb)

        if landmarks:
            self.draw_landmarks(frame, landmarks)

            # Save the image with landmarks to the specified path
            cv2.imwrite(self.save_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Display the captured image
            captured_image = cv2.imread(self.save_path)
            captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
            captured_image = Image.fromarray(captured_image)
            captured_image = ImageTk.PhotoImage(image=captured_image)
            self.captured_image_label.config(image=captured_image)
            self.captured_image_label.image = captured_image

    def draw_landmarks(self, frame, landmarks):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
        )

        # Draw dots on specific landmarks (eyes, lips, etc.)
        for landmark_index in [107, 70, 127, 105, 61, 215, 308, 13, 152, 0, 17, 334, 300, 336, 435]:
            x, y = self.get_landmark_coordinates(landmarks, landmark_index)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    def get_landmark_coordinates(self, landmarks, index):
        x = int(landmarks.landmark[index].x * self.cap.get(3))
        y = int(landmarks.landmark[index].y * self.cap.get(4))
        return x, y

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            landmarks = self.detect_landmarks(frame)
            if landmarks:
                self.draw_landmarks(frame, landmarks)
            
            self.photo = self.convert_frame_to_photo(frame)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.root.after(10, self.update)

    def convert_frame_to_photo(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        return photo

    def detect_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

    def run(self):
        self.root.mainloop()

    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialPointsApp(root, "Facial Points Mapping")
    app.run()
