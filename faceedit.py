import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from scipy.interpolate import griddata

class ImagePinchingTool:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Image Pinching Tool")

        self.original_image = Image.open(image_path).resize((400, 400), Image.LANCZOS)
        self.edited_image = self.original_image.copy()

        self.original_photo = ImageTk.PhotoImage(self.original_image)
        self.edited_photo = ImageTk.PhotoImage(self.edited_image)

        self.canvas = tk.Canvas(root, width=800, height=400)
        self.canvas.pack()

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.original_photo)
        self.canvas.create_image(400, 0, anchor=tk.NW, image=self.edited_photo)

        self.points = []
        self.point_count = 0

        self.canvas.bind("<Button-1>", self.place_point)

        self.pinch_button = tk.Button(root, text="Pinch Image", command=self.pinch_image, state=tk.DISABLED)
        self.pinch_button.pack()

        self.clear_button = tk.Button(root, text="Clear Points", command=self.clear_points)
        self.clear_button.pack()

        self.exit_button = tk.Button(root, text="Exit", command=self.root.destroy)
        self.exit_button.pack()

    def place_point(self, event):
        if self.point_count < 2:
            x, y = event.x, event.y
            self.points.append((x, y))

            # Draw a point on the canvas
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", outline="red")

            self.point_count += 1

            # If two points are placed, enable the pinch button
            if self.point_count == 2:
                self.pinch_button["state"] = tk.NORMAL

    def pinch_image(self):
        if len(self.points) == 2:
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]

            # Generate coordinates for the line between two points
            coords_x, coords_y = np.linspace(x1, x2, num=400), np.linspace(y1, y2, num=400)

            # Create a grid of coordinates
            grid_x, grid_y = np.meshgrid(coords_x, coords_y)

            # Flatten the grid and original image to 1D arrays
            flat_grid_x, flat_grid_y = grid_x.flatten(), grid_y.flatten()
            flat_original_image = np.asarray(self.original_image).reshape(-1, 3)

            # Interpolate the pixel values on the flattened grid
            flat_edited_image = griddata(
                (flat_grid_x, flat_grid_y),
                flat_original_image,
                (flat_grid_x, flat_grid_y),
                method='linear'
            )

            # Reshape the flattened edited image back to 2D
            edited_image_array = flat_edited_image.reshape((400, 400, 3))
            self.edited_image = Image.fromarray(np.uint8(edited_image_array))

            # Update the edited image on the canvas
            self.edited_photo = ImageTk.PhotoImage(self.edited_image)
            
            # Clear the canvas before displaying the new images
            self.canvas.delete("all")
            
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.original_photo)
            self.canvas.create_image(400, 0, anchor=tk.NW, image=self.edited_photo)

            # Disable the pinch button after applying the pinch effect
            self.pinch_button["state"] = tk.DISABLED

    def clear_points(self):
        # Clear points on the canvas
        for point in self.points:
            x, y = point
            self.canvas.delete(self.canvas.find_overlapping(x-6, y-6, x+6, y+6))

        # Reset points and point count
        self.points = []
        self.point_count = 0

        # Redraw the original and edited images on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.original_photo)
        self.canvas.create_image(400, 0, anchor=tk.NW, image=self.edited_photo)

        # Disable the pinch button after clearing points
        self.pinch_button["state"] = tk.DISABLED

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Ask the user to select an image
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

    if file_path:
        app = ImagePinchingTool(tk.Toplevel(), file_path)
        root.mainloop()
