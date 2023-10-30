import cv2
import tkinter as tk
import mediapipe as mp
import math
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

        self.data_label = tk.Label(root, text="Facial Features:")
        self.data_label.grid(row=1, column=0, columnspan=2, pady=10)

        self.btn_capture = tk.Button(root, text="Capture", command=self.capture_image)
        self.btn_capture.grid(row=2, column=0, columnspan=2, pady=10)

        self.update()

        self.save_path = r'D:\face map script\facepics\img2.jpg'  # Update this path to your desired location

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

            # Estimate face data
            face_data = self.estimate_face_data(landmarks)
            self.display_face_data(face_data)

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

    def estimate_face_data(self, landmarks):
        face_data = {}

        # Example: Calculate distances between specific landmarks
        eye_distance = self.calculate_distance(landmarks, 70, 105)
        face_data['Eye Distance'] = eye_distance

        eyebrow_distance = self.calculate_distance(landmarks, 107, 152)
        face_data['Eyebrow Distance'] = eyebrow_distance

        jawline_sharpness = self.calculate_jawline_sharpness(landmarks)
        face_data['Jawline Sharpness'] = jawline_sharpness

        distance_between_eyes = self.calculate_distance(landmarks, 61, 291)
        face_data['Distance Between Eyes'] = distance_between_eyes

        length_of_face = self.calculate_distance(landmarks, 152, 10)
        face_data['Length of Face'] = length_of_face

        # Calculate the golden ratio of the face
        hairline_to_chin = self.calculate_distance(landmarks, 10, 152)
        golden_ratio_face = (hairline_to_chin / eye_distance) * 100
        face_data['Golden Ratio of Face'] = golden_ratio_face

        return face_data

    def calculate_distance(self, landmarks, index1, index2):
        x1, y1 = self.get_landmark_coordinates(landmarks, index1)
        x2, y2 = self.get_landmark_coordinates(landmarks, index2)
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def calculate_jawline_sharpness(self, landmarks):
        # Example: Calculate the angle between specific jawline landmarks (change as needed)
        angle1 = self.calculate_angle(landmarks, 4, 152, 308)
        angle2 = self.calculate_angle(landmarks, 8, 152, 308)
        jawline_sharpness = (angle1 + angle2) / 2
        return jawline_sharpness

    def calculate_angle(self, landmarks, index1, index2, index3):
        x1, y1 = self.get_landmark_coordinates(landmarks, index1)
        x2, y2 = self.get_landmark_coordinates(landmarks, index2)
        x3, y3 = self.get_landmark_coordinates(landmarks, index3)
        angle_rad = math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def display_face_data(self, face_data):
        data_text = "\n".join([f"{key}: {value}" for key, value in face_data.items()])
        self.data_label.config(text=f"Facial Features:\n{data_text}")

    def run(self):
        self.root.mainloop()

    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialPointsApp(root, "Facial Points Mapping")
    app.run()
