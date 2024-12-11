import numpy as np
from PIL import Image, ImageDraw
import json
import os

class ImageProcessor:
    def __init__(self, image_path):
        """
        Initialize the ImageProcessor with the given image.

        Parameters:
        - image_path (str): Path to the image file.
        """
        self.image = Image.open(image_path).convert('RGB')
        self.labels = np.full((self.image.height, self.image.width), 128, dtype=np.uint8)  # 128: Unidentified
        self.history = []  # Stack for undo
        self.redo_stack = []  # Stack for redo

    def save_state(self):
        """
        Save the current state of labels to the history stack.
        Clears the redo stack as new actions invalidate the redo history.
        """
        self.history.append(self.labels.copy())
        self.redo_stack.clear()

    def undo(self):
        """
        Undo the last action by reverting to the previous label state.

        Returns:
        - success (bool): True if undo was successful, False otherwise.
        """
        if not self.history:
            print("Undo stack is empty.")
            return False
        self.redo_stack.append(self.labels.copy())
        self.labels = self.history.pop()
        return True

    def redo(self):
        """
        Redo the last undone action by restoring from the redo stack.

        Returns:
        - success (bool): True if redo was successful, False otherwise.
        """
        if not self.redo_stack:
            print("Redo stack is empty.")
            return False
        self.history.append(self.labels.copy())
        self.labels = self.redo_stack.pop()
        return True

    def mark_foreground(self, coordinates, brush_size=10):
        """
        Mark pixels around the given coordinates as foreground.

        Parameters:
        - coordinates (list of tuples): List of (x, y) tuples indicating points to mark.
        - brush_size (int): Radius around each point to mark.
        """
        self.save_state()  # Save current state before making changes
        for (x, y) in coordinates:
            self._draw_circle(x, y, brush_size, label=255)

    def mark_background(self, coordinates, brush_size=10):
        """
        Mark pixels around the given coordinates as background.

        Parameters:
        - coordinates (list of tuples): List of (x, y) tuples indicating points to mark.
        - brush_size (int): Radius around each point to mark.
        """
        self.save_state()  # Save current state before making changes
        for (x, y) in coordinates:
            self._draw_circle(x, y, brush_size, label=0)

    def _draw_circle(self, x_center, y_center, radius, label):
        """
        Helper method to draw a filled circle on the labels array.

        Parameters:
        - x_center (int): X-coordinate of the circle center.
        - y_center (int): Y-coordinate of the circle center.
        - radius (int): Radius of the circle.
        - label (int): Label value to set (0 for background, 255 for foreground).
        """
        # Define the bounding box of the circle
        x_min = max(x_center - radius, 0)
        x_max = min(x_center + radius, self.image.width - 1)
        y_min = max(y_center - radius, 0)
        y_max = min(y_center + radius, self.image.height - 1)

        # Create a grid of coordinates within the bounding box
        y, x = np.ogrid[y_min:y_max+1, x_min:x_max+1]
        distance = (x - x_center) ** 2 + (y - y_center) ** 2
        mask = distance <= radius ** 2

        # Update labels where mask is True and pixels are not locked (assuming no locks for now)
        self.labels[y_min:y_max+1, x_min:x_max+1][mask] = label

    def get_processed_image(self):
        """
        Generate an RGB image based on the current labels.

        Returns:
        - processed_image (PIL.Image.Image): Image representing the labels.
        """
        # Create an RGB image from labels
        processed_rgb = np.zeros((self.labels.shape[0], self.labels.shape[1], 3), dtype=np.uint8)

        # Define colors for each label
        # 0: Black (Background), 128: Gray (Unidentified), 255: White (Foreground)
        processed_rgb[self.labels == 0] = [0, 0, 0]         # Background
        processed_rgb[self.labels == 128] = [128, 128, 128]  # Unidentified
        processed_rgb[self.labels == 255] = [255, 255, 255]  # Foreground

        # Convert NumPy array to PIL Image
        processed_image = Image.fromarray(processed_rgb, mode='RGB')
        return processed_image

    def save_labels(self, save_path):
        """
        Save the current labels to a file.

        Parameters:
        - save_path (str): Path where the labels will be saved (.npy format).
        """
        np.save(save_path, self.labels)
        print(f"Labels saved to {save_path}")

    def load_labels(self, load_path):
        """
        Load labels from a file.

        Parameters:
        - load_path (str): Path from where the labels will be loaded (.npy format).
        """
        if not os.path.exists(load_path):
            print(f"Label file {load_path} does not exist.")
            return
        self.labels = np.load(load_path)
        print(f"Labels loaded from {load_path}")

    def reset_labels(self):
        """
        Reset all labels to unidentified (128).
        """
        self.save_state()  # Save current state before resetting
        self.labels.fill(128)
        print("All labels have been reset to unidentified.")

    def export_label_image(self, save_path):
        """
        Export the label image as a grayscale PNG for visualization.

        Parameters:
        - save_path (str): Path where the label image will be saved (.png format).
        """
        # Map labels to grayscale values
        label_image = Image.fromarray(self.labels, mode='L')  # 'L' mode for grayscale
        label_image.save(save_path)
        print(f"Label image exported to {save_path}")

    def apply_threshold(self, image_array, threshold_a, threshold_b):
        """
        Apply thresholding to an image array to generate labels.

        Parameters:
        - image_array (numpy.ndarray): Image data as a NumPy array.
        - threshold_a (float): Lower threshold for background.
        - threshold_b (float): Upper threshold for foreground.
        """
        self.save_state()  # Save current state before applying thresholds

        mask_background = image_array < threshold_a
        mask_foreground = image_array > threshold_b
        mask_unidentified = (image_array >= threshold_a) & (image_array <= threshold_b)

        self.labels[mask_background] = 0
        self.labels[mask_foreground] = 255
        self.labels[mask_unidentified] = 128

        print("Thresholding applied to image.")
