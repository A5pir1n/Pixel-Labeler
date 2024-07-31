import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import certifi
import os
import tkinter as tk
import json
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageOps
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class ListNode:
    def __init__(self, state):
        self.state = state
        self.prev = None
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.current = None
        self.size = 0
    
    def append(self, state):
        new_node = ListNode(state)
        
        # If we're not at the end of the list, remove all the nodes after the current one
        if self.current and self.current.next:
            self.current.next = None
            self.tail = self.current
            self.size -= 1
            while self.tail.next:
                self.size -= 1
                self.tail = self.tail.next

        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        self.current = new_node  # Update the current to the new node
        self.size += 1

    def undo(self):
        if self.current and self.current.prev:
            self.current = self.current.prev
            return self.current.state
        return None

    def redo(self):
        if self.current and self.current.next:
            self.current = self.current.next
            return self.current.state
        return None


class ImageLabeler:
    def __init__(self, root, block_size=100):
        self.root = root
        self.root.title("Image Labeler")
        self.block_size = block_size
        self.foreground_pixels = set()
        self.background_pixels = set()
        self.unidentified_pixels = set()
        self.image_path = None
        self.image = None
        self.processed_image = None
        self.custom_model = None

        self.modification_mode = False
        self.temp_foreground_pixels = set()
        self.temp_background_pixels = set()

        self.canvas = tk.Canvas(root)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.processed_canvas = tk.Canvas(root)
        self.processed_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create a new top-level window for the control panel
        self.control_panel = tk.Toplevel(self.root)
        self.control_panel.title("Control Panel")
        self.control_panel.attributes('-topmost', 1)  # Make the control panel always on top

        self.load_image_button = tk.Button(self.control_panel, text="Load Image", command=self.load_image)
        self.load_image_button.pack(side=tk.BOTTOM)

        self.save_labels_button = tk.Button(self.control_panel, text="Save Labels", command=self.save_labels)
        self.save_labels_button.pack(side=tk.BOTTOM)

        self.load_labels_button = tk.Button(self.control_panel, text="Load Labels", command=self.load_labels)
        self.load_labels_button.pack(side=tk.BOTTOM)

        self.clear_button = tk.Button(self.control_panel, text="Clear All", command=self.clear_all)
        self.clear_button.pack(side=tk.BOTTOM)

        self.classify_button = tk.Button(self.control_panel, text="Classify by SAM", command=self.classify_by_sam)
        self.classify_button.pack(side=tk.BOTTOM)

        self.load_custom_model_button = tk.Button(self.control_panel, text="Load Custom Model", command=self.load_custom_model)
        self.load_custom_model_button.pack(side=tk.BOTTOM)

        self.undo_button = tk.Button(self.control_panel, text="Undo", command=self.undo)
        self.undo_button.pack(side=tk.BOTTOM)

        self.redo_button = tk.Button(self.control_panel, text="Redo", command=self.redo)
        self.redo_button.pack(side=tk.BOTTOM)

        self.threshold_a_label = tk.Label(self.control_panel, text="Threshold A (Background):")
        self.threshold_a_label.pack(side=tk.BOTTOM)
        self.threshold_a = tk.Entry(self.control_panel)
        self.threshold_a.pack(side=tk.BOTTOM)

        self.threshold_b_label = tk.Label(self.control_panel, text="Threshold B (Foreground):")
        self.threshold_b_label.pack(side=tk.BOTTOM)
        self.threshold_b = tk.Entry(self.control_panel)
        self.threshold_b.pack(side=tk.BOTTOM)

        self.lambda_label = tk.Label(self.control_panel, text="Lambda Value:")
        self.lambda_label.pack(side=tk.BOTTOM)
        self.lambda_slider = tk.Scale(self.control_panel, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, command=self.update_display)
        self.lambda_slider.pack(side=tk.BOTTOM)

        self.tolerance_label = tk.Label(self.control_panel, text="Tolerance Level:")
        self.tolerance_label.pack(side=tk.BOTTOM)
        self.tolerance_slider = tk.Scale(self.control_panel, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL, command=self.update_display_in_modification_mode)
        self.tolerance_slider.set(30)  # Default value
        self.tolerance_slider.pack(side=tk.BOTTOM)

        self.marking_mode = tk.StringVar(value="foreground")
        self.foreground_checkbox = tk.Radiobutton(self.control_panel, text="Marking Foreground", variable=self.marking_mode, value="foreground")
        self.background_checkbox = tk.Radiobutton(self.control_panel, text="Marking Background", variable=self.marking_mode, value="background")
        self.foreground_checkbox.pack(side=tk.TOP, anchor=tk.W)
        self.background_checkbox.pack(side=tk.TOP, anchor=tk.W)

        self.marking_style = tk.StringVar(value="circled")
        self.circled_area_checkbox = tk.Radiobutton(self.control_panel, text="Marking on Circled Areas", variable=self.marking_style, value="circled")
        self.dragged_area_checkbox = tk.Radiobutton(self.control_panel, text="Marking on Dragged Areas", variable=self.marking_style, value="dragged")
        self.circled_area_checkbox.pack(side=tk.TOP, anchor=tk.W)
        self.dragged_area_checkbox.pack(side=tk.TOP, anchor=tk.W)

        self.marking_scope = tk.StringVar(value="local")
        self.local_mode_button = tk.Radiobutton(self.control_panel, text="Local Mode", variable=self.marking_scope, value="local")
        self.global_mode_button = tk.Radiobutton(self.control_panel, text="Global Mode", variable=self.marking_scope, value="global")
        self.local_mode_button.pack(side=tk.TOP, anchor=tk.W)
        self.global_mode_button.pack(side=tk.TOP, anchor=tk.W)

        self.drawing = False
        self.drawn_lines = []
        self.single_click_position = None
        self.moved = False

        self.history = LinkedList()
        self.save_state()

        MAC_OS = False
        if sys.platform == 'darwin':
            MAC_OS = True
        if MAC_OS:
            self.canvas.bind("<Button-2>", self.on_right_click)
        else:
            self.canvas.bind("<Button-3>", self.on_right_click)

        self.canvas.bind("<Button-1>", self.start_draw_or_click)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw_or_click)

    def count_states(self):
        current = self.history.head
        count = 0
        while current:
            count += 1
            current = current.next
        return count

    def save_state(self):
        state = {
            "foreground_pixels": self.foreground_pixels.copy(),
            "background_pixels": self.background_pixels.copy(),
            "unidentified_pixels": self.unidentified_pixels.copy()
        }
        self.history.append(state)

    def load_state(self, state):
        self.foreground_pixels = state["foreground_pixels"]
        self.background_pixels = state["background_pixels"]
        self.unidentified_pixels = state["unidentified_pixels"]

        for tag in self.canvas.find_withtag('shade_'):
            self.canvas.delete(tag)

        # Re-apply shades based on the current state
        for pixel in self.foreground_pixels:
            block_x = pixel[0] // self.block_size
            block_y = pixel[1] // self.block_size
            self.shade_block(block_x, block_y, 'red')

        for pixel in self.background_pixels:
            block_x = pixel[0] // self.block_size
            block_y = pixel[1] // self.block_size
            self.shade_block(block_x, block_y, 'black')

        self.update_processed_image()

    def undo(self):
        state = self.history.undo()
        if state:
            self.load_state(state)

    def redo(self):
        state = self.history.redo()
        if state:
            self.load_state(state)

    def enter_modification_mode(self):
        self.modification_mode = True
        self.temp_foreground_pixels = self.foreground_pixels.copy()
        self.temp_background_pixels = self.background_pixels.copy()

        self.apply_button = tk.Button(self.control_panel, text="Apply", command=self.apply_changes)
        self.apply_button.pack(side=tk.BOTTOM)
        self.cancel_button = tk.Button(self.control_panel, text="Cancel", command=self.cancel_changes)
        self.cancel_button.pack(side=tk.BOTTOM)

    def exit_modification_mode(self):
        self.modification_mode = False
        self.apply_button.pack_forget()
        self.cancel_button.pack_forget()

    def apply_changes(self):
        self.foreground_pixels.update(self.temp_foreground_pixels)
        self.background_pixels.update(self.temp_background_pixels)
        self.unidentified_pixels -= self.temp_foreground_pixels
        self.unidentified_pixels -= self.temp_background_pixels
        self.save_state() 
        self.exit_modification_mode()
        self.update_processed_image()
        


    def cancel_changes(self):
        self.exit_modification_mode()
        self.update_processed_image()
        
    def update_display_in_modification_mode(self, event):
        if self.modification_mode:
            tolerance = self.tolerance_slider.get()
            target_color = self.reference_color

            if target_color:
                self.processed_image = Image.new("RGB", self.image.size, "black")
                width, height = self.image.size
                for pixel_x in range(width):
                    for pixel_y in range(height):
                        current_color = self.image.getpixel((pixel_x, pixel_y))
                        if all(abs(current_color[k] - target_color[k]) < tolerance for k in range(3)):
                            if self.marking_mode.get() == "foreground":
                                self.processed_image.putpixel((pixel_x, pixel_y), (255, 255, 255))  # White for foreground
                                self.temp_foreground_pixels.add((pixel_x, pixel_y))
                                self.temp_background_pixels.discard((pixel_x, pixel_y))
                            elif self.marking_mode.get() == "background":
                                self.processed_image.putpixel((pixel_x, pixel_y), (0, 0, 0))  # Black for background
                                self.temp_background_pixels.add((pixel_x, pixel_y))
                                self.temp_foreground_pixels.discard((pixel_x, pixel_y))
                        else:
                            self.processed_image.putpixel((pixel_x, pixel_y), (128, 128, 128))  # Gray for unidentified

                self.processed_photo = ImageTk.PhotoImage(self.processed_image)
                self.processed_canvas.create_image(0, 0, image=self.processed_photo, anchor=tk.NW)


    def load_custom_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
        if not model_path:
            return

        self.custom_model = torch.load(model_path)
        self.custom_model.eval()
        print(f"Custom model loaded from {model_path}")

    def classify_by_sam(self):
        # Load the pre-trained Mask R-CNN model if no custom model is loaded
        if self.custom_model:
            model = self.custom_model
        else:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
            model.eval()

        # Load and preprocess the image
        img = Image.open(self.image_path).convert("RGB")
        img = np.array(img)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        img_tensor = transform(img)

        # Perform inference
        with torch.no_grad():
            predictions = model([img_tensor])[0]

        # Extract the masks
        masks = predictions['masks'].cpu().numpy()
        combined_mask = np.zeros_like(masks[0, 0])

        for mask in masks:
            combined_mask = np.maximum(combined_mask, mask[0])

        # Get thresholds from the user inputs
        try:
            threshold_a = float(self.threshold_a.get())
            threshold_b = float(self.threshold_b.get())
        except ValueError:
            print("Invalid thresholds. Please enter numeric values.")
            return

        # Classify pixels based on thresholds
        height, width = combined_mask.shape
        for y in range(height):
            for x in range(width):
                pixel_value = combined_mask[y, x]
                if pixel_value < threshold_a:
                    self.background_pixels.add((x, y))
                    self.processed_image.putpixel((x, y), (0, 0, 0))  # Black for background
                elif pixel_value > threshold_b:
                    self.foreground_pixels.add((x, y))
                    self.processed_image.putpixel((x, y), (255, 255, 255))  # White for foreground
                else:
                    self.unidentified_pixels.add((x, y))
                    self.processed_image.putpixel((x, y), (128, 128, 128))  # Gray for unidentified

        # Update the processed image
        self.processed_photo = ImageTk.PhotoImage(self.processed_image)
        self.processed_canvas.create_image(0, 0, image=self.processed_photo, anchor=tk.NW)
        self.update_processed_image()


    def load_labels(self):
        label_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not label_path:
            return

        with open(label_path, 'r') as f:
            labels = json.load(f)

        self.foreground_pixels = set(tuple(pixel) for pixel in labels["foreground"])
        self.background_pixels = set(tuple(pixel) for pixel in labels["background"])
        self.unidentified_pixels = set(
            (x, y) for x in range(self.image.width) for y in range(self.image.height)
        ) - self.foreground_pixels - self.background_pixels

        self.update_processed_image()
        print(f"Labels loaded from {label_path}")
        
    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.image = Image.open(self.image_path)
            original_width, original_height = self.image.size

            # Calculate the new width and height to be multiples of 100
            self.new_width = ((original_width + 99) // 100) * 100
            self.new_height = ((original_height + 99) // 100) * 100

            # Create a new image with the new size and a white background
            new_image = Image.new("RGB", (self.new_width, self.new_height), "white")
            new_image.paste(self.image, (0, 0))

            self.image = new_image
            self.photo = ImageTk.PhotoImage(self.image)
            self.processed_image = Image.new("RGB", self.image.size, "grey")
            self.processed_photo = ImageTk.PhotoImage(self.processed_image)
            self.canvas.config(width=self.photo.width(), height=self.photo.height())
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.processed_canvas.config(width=self.processed_photo.width(), height=self.processed_photo.height())
            self.processed_canvas.create_image(0, 0, image=self.processed_photo, anchor=tk.NW)
            self.draw_grid()

            # Store the original dimensions
            self.original_width = original_width
            self.original_height = original_height

            # Initialize all pixels as unidentified
            for x in range(self.new_width):
                for y in range(self.new_height):
                    self.unidentified_pixels.add((x, y))


    def clear_all(self):
        self.foreground_pixels.clear()
        self.background_pixels.clear()
        self.unidentified_pixels.clear()

        # Reset all pixels to unidentified
        for x in range(self.new_width):
            for y in range(self.new_height):
                self.unidentified_pixels.add((x, y))

        # Clear the canvas
        self.canvas.delete("all")
        self.processed_canvas.delete("all")

        # Redraw the image and grid
        if self.image:
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.processed_image = Image.new("RGB", self.image.size, "grey")
            self.processed_photo = ImageTk.PhotoImage(self.processed_image)
            self.processed_canvas.create_image(0, 0, image=self.processed_photo, anchor=tk.NW)
            self.draw_grid()
        self.save_state()
         

    def update_display(self, event=None):
        if self.image is None or self.processed_image is None:
            return

        lambda_value = self.lambda_slider.get()
        blended_image = Image.new("RGB", self.image.size)
        
        original_pixels = self.image.load()
        processed_pixels = self.processed_image.load()
        blended_pixels = blended_image.load()

        width, height = self.image.size
        for y in range(height):
            for x in range(width):
                r, g, b = original_pixels[x, y]
                r_prime, g_prime, b_prime = processed_pixels[x, y]
                blended_r = int(lambda_value * r + (1 - lambda_value) * r_prime)
                blended_g = int(lambda_value * g + (1 - lambda_value) * g_prime)
                blended_b = int(lambda_value * b + (1 - lambda_value) * b_prime)
                blended_pixels[x, y] = (blended_r, blended_g, blended_b)

        self.processed_photo = ImageTk.PhotoImage(blended_image)
        self.processed_canvas.create_image(0, 0, image=self.processed_photo, anchor=tk.NW)


    def draw_grid(self):
        for i in range(0, self.photo.width(), self.block_size):
            self.canvas.create_line([(i, 0), (i, self.photo.height())], fill='white', tag='grid_line')
        for i in range(0, self.photo.height(), self.block_size):
            self.canvas.create_line([(0, i), (self.photo.width(), i)], fill='white', tag='grid_line')

    def start_draw_or_click(self, event):
        self.drawing = False
        self.single_click_position = (event.x, event.y)
        self.moved = False
        self.drawn_lines = [[event.x, event.y]]

    def draw(self, event):
        self.moved = True
        if not self.drawing:
            self.drawing = True
            self.drawn_lines = [[event.x, event.y]]

        self.drawn_lines.append([event.x, event.y])
        self.canvas.create_line(
            self.drawn_lines[-2][0], self.drawn_lines[-2][1],
            self.drawn_lines[-1][0], self.drawn_lines[-1][1],
            fill='blue',
            tags='blue_line'
        )

        if self.marking_style.get() == "dragged":
            self.process_dragged_line(event.x, event.y)


    def process_dragged_line(self, x, y):
        if len(self.drawn_lines) < 2:
            return

        x1, y1 = self.drawn_lines[-2]
        x2, y2 = self.drawn_lines[-1]

        # Interpolate points between (x1, y1) and (x2, y2)
        num_points = max(abs(x2 - x1), abs(y2 - y1))  # Number of points to interpolate
        num_points = max(num_points, 1)  # Ensure num_points is at least 1 to avoid division by zero

        for i in range(num_points + 1):
            xi = x1 + i * (x2 - x1) / num_points
            yi = y1 + i * (y2 - y1) / num_points
            block_x = int(xi) // self.block_size
            block_y = int(yi) // self.block_size

            if self.marking_mode.get() == "foreground":
                if not self.is_block_foreground(block_x, block_y):
                    self.add_block_pixels_to_foreground(block_x, block_y)
                    self.canvas.create_rectangle(
                        block_x * self.block_size, block_y * self.block_size,
                        (block_x + 1) * self.block_size, (block_y + 1) * self.block_size,
                        outline='red', fill='', width=2
                    )
            elif self.marking_mode.get() == "background":
                if not self.is_block_background(block_x, block_y):
                    self.add_block_pixels_to_background(block_x, block_y)
                    self.canvas.create_rectangle(
                        block_x * self.block_size, block_y * self.block_size,
                        (block_x + 1) * self.block_size, (block_y + 1) * self.block_size,
                        outline='black', fill='', width=2
                    )

    def end_draw_or_click(self, event):
        if self.moved and self.drawing:
            self.drawn_lines.append([event.x, event.y])
            if self.marking_style.get() == 'circled':
                self.process_drawn_area(self.canvas)
        else:
            self.process_single_click(event.x, event.y)
        self.drawing = False
        self.single_click_position = None
        self.canvas.delete('blue_line')
        self.save_state() 
        self.update_processed_image()


    def is_block_foreground(self, x, y, block_size=None):
        if block_size is None:
            block_size = self.block_size
        for i in range(x * block_size, (x + 1) * block_size):
            for j in range(y * block_size, (y + 1) * block_size):
                if (i, j) not in self.foreground_pixels:
                    return False
        return True

    def is_block_background(self, x, y, block_size=None):
        if block_size is None:
            block_size = self.block_size
        for i in range(x * block_size, (x + 1) * block_size):
            for j in range(y * block_size, (y + 1) * block_size):
                if (i, j) not in self.background_pixels:
                    return False
        return True

    def add_block_pixels_to_foreground(self, x, y, block_size=None):
        if block_size is None:
            block_size = self.block_size
        for i in range(x * block_size, (x + 1) * block_size):
            for j in range(y * block_size, (y + 1) * block_size):
                self.foreground_pixels.add((i, j))
                self.unidentified_pixels.discard((i, j))
                self.background_pixels.discard((i, j))

    def remove_block_pixels_from_foreground(self, x, y, block_size=None):
        if block_size is None:
            block_size = self.block_size
        for i in range(x * block_size, (x + 1) * block_size):
            for j in range(y * block_size, (y + 1) * block_size):
                self.foreground_pixels.discard((i, j))
                self.unidentified_pixels.add((i, j))

    def add_block_pixels_to_background(self, x, y, block_size=None):
        if block_size is None:
            block_size = self.block_size
        for i in range(x * block_size, (x + 1) * block_size):
            for j in range(y * block_size, (y + 1) * block_size):
                self.background_pixels.add((i, j))
                self.unidentified_pixels.discard((i, j))
                self.foreground_pixels.discard((i, j))

    def remove_block_pixels_from_background(self, x, y, block_size=None):
        if block_size is None:
            block_size = self.block_size
        for i in range(x * block_size, (x + 1) * block_size):
            for j in range(y * block_size, (y + 1) * block_size):
                self.background_pixels.discard((i, j))
                self.unidentified_pixels.add((i, j))


    def process_single_click(self, x, y):
        block_x = x // self.block_size
        block_y = y // self.block_size

        if self.marking_mode.get() == "foreground":
            if self.is_block_foreground(block_x, block_y):
                self.remove_block_pixels_from_foreground(block_x, block_y)
                self.clear_shade(block_x, block_y)
                self.redraw_block(block_x, block_y)
            else:
                self.add_block_pixels_to_foreground(block_x, block_y)
                self.shade_block(block_x, block_y, 'red')

        elif self.marking_mode.get() == "background":
            if self.is_block_background(block_x, block_y):
                self.remove_block_pixels_from_background(block_x, block_y)
                self.clear_shade(block_x, block_y)
                self.redraw_block(block_x, block_y)
            else:
                self.add_block_pixels_to_background(block_x, block_y)
                self.shade_block(block_x, block_y, 'black')

        self.update_processed_image()

    def shade_block(self, block_x, block_y, color):
        x1 = block_x * self.block_size
        y1 = block_y * self.block_size
        x2 = x1 + self.block_size
        y2 = y1 + self.block_size

        num_lines = 6
        line_spacing = self.block_size // num_lines
        tag = f'shade_{block_x}_{block_y}'

        for i in range(1, num_lines):
            self.canvas.create_line(x1, y2 - i * line_spacing, x1 + i * line_spacing, y2, fill=color, width=2, tags=tag)
            self.canvas.create_line(x2 - i * line_spacing, y1, x2, y1 + i * line_spacing, fill=color, width=2, tags=tag)
        
        # Draw the middle diagonal line
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2, tags=tag)

    def clear_shade(self, block_x, block_y):
        tag = f'shade_{block_x}_{block_y}'
        self.canvas.delete(tag)

    def process_drawn_area(self, canvas):
        if len(self.drawn_lines) < 3:
            canvas.delete('blue_line')
            self.drawn_lines = []
            return

        enclosed_blocks = self.get_enclosed_blocks(self.drawn_lines)
        affected_pixels = []

        if self.marking_mode.get() == "foreground":
            for block_id in enclosed_blocks:
                block_x, block_y = block_id
                if self.is_block_foreground(block_x, block_y):
                    self.remove_block_pixels_from_foreground(block_x, block_y)
                    self.clear_shade(block_x, block_y)
                else:
                    self.add_block_pixels_to_foreground(block_x, block_y)
                    self.shade_block(block_x, block_y, 'red')
                    affected_pixels.append((block_x, block_y, "background"))

        elif self.marking_mode.get() == "background":
            for block_id in enclosed_blocks:
                block_x, block_y = block_id
                if self.is_block_background(block_x, block_y):
                    self.remove_block_pixels_from_background(block_x, block_y)
                    self.clear_shade(block_x, block_y)
                else:
                    self.add_block_pixels_to_background(block_x, block_y)
                    self.shade_block(block_x, block_y, 'black')
                    affected_pixels.append((block_x, block_y, "foreground"))

        self.update_processed_image()
        canvas.delete('blue_line')
        self.drawn_lines = []
        # self.save_state()



    def get_enclosed_blocks(self, points):
        enclosed_blocks = set()
        for i in range(0, self.photo.width(), self.block_size):
            for j in range(0, self.photo.height(), self.block_size):
                if self.is_point_in_polygon(i, j, points) or self.is_point_in_polygon(i + self.block_size - 1, j, points) or self.is_point_in_polygon(i, j + self.block_size - 1, points) or self.is_point_in_polygon(i + self.block_size - 1, j + self.block_size - 1, points):
                    enclosed_blocks.add((i // self.block_size, j // self.block_size))
        return enclosed_blocks

    def is_point_in_polygon(self, x, y, poly):
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def on_right_click(self, event):
        x = event.x // self.block_size
        y = event.y // self.block_size
        block_id = (x, y)
        self.open_detailed_window(block_id)

    def open_detailed_window(self, block_id):
        detailed_window = tk.Toplevel(self.root)
        detailed_window.title(f"Detailed View: Block {block_id}")
        detailed_canvas = tk.Canvas(detailed_window)
        detailed_canvas.pack(fill=tk.BOTH, expand=True)
        
        block_image = self.image.crop(
            (block_id[0] * self.block_size, block_id[1] * self.block_size, 
             (block_id[0] + 1) * self.block_size, (block_id[1] + 1) * self.block_size)
        )
        detailed_image = block_image.resize((self.block_size * 10, self.block_size * 10), Image.LANCZOS)
        detailed_photo = ImageTk.PhotoImage(detailed_image)
        
        detailed_canvas.config(width=detailed_photo.width(), height=detailed_photo.height())
        detailed_canvas.create_image(0, 0, image=detailed_photo, anchor=tk.NW)
        
        # Draw the smaller grid and mark foreground pixels
        small_block_size = detailed_photo.width() // 10
        for i in range(0, detailed_photo.width(), small_block_size):
            for j in range(0, detailed_photo.height(), small_block_size):
                pixel_id = (block_id[0] * self.block_size + (i // small_block_size) * (self.block_size // 10),
                            block_id[1] * self.block_size + (j // small_block_size) * (self.block_size // 10))
                if pixel_id in self.foreground_pixels:
                    detailed_canvas.create_rectangle(
                        i, j,
                        i + small_block_size, j + small_block_size,
                        outline='red', fill='', width=2
                    )
                elif pixel_id in self.background_pixels:
                    detailed_canvas.create_rectangle(
                        i, j,
                        i + small_block_size, j + small_block_size,
                        outline = 'black', fill = '', width = 2
                    )
                else:
                    detailed_canvas.create_line([(i, j), (i, j + small_block_size)], fill='white')
                    detailed_canvas.create_line([(i, j), (i + small_block_size, j)], fill='white')
                    detailed_canvas.create_line([(i + small_block_size, j), (i + small_block_size, j + small_block_size)], fill='white')
                    detailed_canvas.create_line([(i, j + small_block_size), (i + small_block_size, j + small_block_size)], fill='white')

        detailed_canvas.bind("<Button-1>", lambda event: self.start_detailed_draw_or_click(event, detailed_canvas, block_id, small_block_size))
        detailed_canvas.bind("<Button-2>", lambda event: self.on_detailed_right_click(event, detailed_canvas, block_id, small_block_size))
        detailed_canvas.bind("<B1-Motion>", lambda event: self.draw_detailed(event, detailed_canvas, block_id, small_block_size))
        detailed_canvas.bind("<ButtonRelease-1>", lambda event: self.end_detailed_draw_or_click(event, detailed_canvas, block_id, small_block_size))

        # Store the image reference to avoid garbage collection
        detailed_canvas.image = detailed_photo

    def on_detailed_right_click(self, event, canvas, block_id, small_block_size):
        x = event.x // small_block_size
        y = event.y // small_block_size
        detailed_block_id = (x, y)
        self.open_third_level_window(block_id, detailed_block_id)

    def start_detailed_draw_or_click(self, event, canvas, block_id, small_block_size):
        self.drawing = False
        self.single_click_position = (event.x, event.y)
        self.moved = False
        self.drawn_lines = [[event.x, event.y]]

    def draw_detailed(self, event, canvas, block_id, small_block_size):
        self.moved = True
        if not self.drawing:
            self.drawing = True
            self.drawn_lines = [[event.x, event.y]]

        self.drawn_lines.append([event.x, event.y])
        canvas.create_line(
            self.drawn_lines[-2][0], self.drawn_lines[-2][1],
            self.drawn_lines[-1][0], self.drawn_lines[-1][1],
            fill='blue',
            tags='blue_line'
        )

        if self.marking_style.get() == "dragged":
            self.process_dragged_line_detailed(event.x, event.y, canvas, block_id, small_block_size)

    def process_dragged_line_detailed(self, x, y, canvas, block_id, small_block_size):
        if len(self.drawn_lines) < 2:
            return

        x1, y1 = self.drawn_lines[-2]
        x2, y2 = self.drawn_lines[-1]

        # Interpolate points between (x1, y1) and (x2, y2)
        num_points = max(abs(x2 - x1), abs(y2 - y1))  # Number of points to interpolate
        num_points = max(num_points, 1)  # Ensure num_points is at least 1 to avoid division by zero

        detailed_block_size = self.block_size // 10
        for i in range(num_points + 1):
            xi = x1 + i * (x2 - x1) / num_points
            yi = y1 + i * (y2 - y1) / num_points
            block_x = block_id[0] * detailed_block_size + int(xi // small_block_size)
            block_y = block_id[1] * detailed_block_size + int(yi // small_block_size)

            grid_x = (block_x % detailed_block_size)
            grid_y = (block_y % detailed_block_size)

            if self.marking_mode.get() == "foreground":
                if not self.is_block_foreground(block_x, block_y, detailed_block_size):
                    self.add_block_pixels_to_foreground(block_x, block_y, detailed_block_size)
                    canvas.create_rectangle(
                        grid_x * small_block_size,
                        grid_y * small_block_size,
                        (grid_x + 1) * small_block_size,
                        (grid_y + 1) * small_block_size,
                        outline='red', fill='', width=2
                    )
            elif self.marking_mode.get() == "background":
                if not self.is_block_background(block_x, block_y, detailed_block_size):
                    self.add_block_pixels_to_background(block_x, block_y, detailed_block_size)
                    canvas.create_rectangle(
                        grid_x * small_block_size,
                        grid_y * small_block_size,
                        (grid_x + 1) * small_block_size,
                        (grid_y + 1) * small_block_size,
                        outline='black', fill='', width=2
                    )

    def end_detailed_draw_or_click(self, event, canvas, block_id, small_block_size):
        if self.moved and self.drawing:
            self.drawn_lines.append([event.x, event.y])
            if self.marking_style.get() == "circled":
                self.process_detailed_drawn_area(canvas, block_id, small_block_size)
            self.drawing = False
            self.single_click_position = None
            canvas.delete('blue_line')
        else:
            self.process_detailed_single_click(event.x, event.y, canvas, block_id, small_block_size)
        self.save_state()
        self.update_processed_image()


    def process_detailed_single_click(self, x, y, canvas, block_id, small_block_size):
        detailed_x = block_id[0] * (self.block_size // 10) + (x // small_block_size)
        detailed_y = block_id[1] * (self.block_size // 10) + (y // small_block_size)

        if self.marking_mode.get() == "foreground":
            if self.is_block_foreground(detailed_x, detailed_y, self.block_size // 10):
                self.remove_block_pixels_from_foreground(detailed_x, detailed_y, self.block_size // 10)
                self.redraw_detailed_block(x // small_block_size, y // small_block_size, canvas, small_block_size)
            else:
                self.add_block_pixels_to_foreground(detailed_x, detailed_y, self.block_size // 10)
                canvas.create_rectangle(
                    (x // small_block_size) * small_block_size, (y // small_block_size) * small_block_size,
                    ((x // small_block_size) + 1) * small_block_size, ((y // small_block_size) + 1) * small_block_size,
                    outline='red', fill='', width=2
                )
        elif self.marking_mode.get() == "background":
            if self.is_block_background(detailed_x, detailed_y, self.block_size // 10):
                self.remove_block_pixels_from_background(detailed_x, detailed_y, self.block_size // 10)
                self.redraw_detailed_block(x // small_block_size, y // small_block_size, canvas, small_block_size)
            else:
                self.add_block_pixels_to_background(detailed_x, detailed_y, self.block_size // 10)
                canvas.create_rectangle(
                    (x // small_block_size) * small_block_size, (y // small_block_size) * small_block_size,
                    ((x // small_block_size) + 1) * small_block_size, ((y // small_block_size) + 1) * small_block_size,
                    outline='black', fill='', width=2
                )
        self.update_processed_image()
        
    def process_detailed_drawn_area(self, canvas, block_id, small_block_size):
        if len(self.drawn_lines) < 3:
            canvas.delete('blue_line')
            self.drawn_lines = []
            return

        enclosed_blocks = self.get_detailed_enclosed_blocks(self.drawn_lines, block_id, small_block_size)
        base_x = block_id[0] * self.block_size
        base_y = block_id[1] * self.block_size
        detailed_block_size = self.block_size // 10

        if self.marking_mode.get() == "foreground":
            for pixel_id in enclosed_blocks:
                detailed_x = pixel_id[0]
                detailed_y = pixel_id[1]
                block_x = base_x + (detailed_x * detailed_block_size)
                block_y = base_y + (detailed_y * detailed_block_size)
                if self.is_block_foreground(detailed_x, detailed_y, detailed_block_size):
                    self.remove_block_pixels_from_foreground(detailed_x, detailed_y, detailed_block_size)
                    self.redraw_detailed_block(detailed_x, detailed_y, canvas, small_block_size)
                else:
                    self.add_block_pixels_to_foreground(detailed_x, detailed_y, detailed_block_size)
                    canvas.create_rectangle(
                        (detailed_x % 10) * small_block_size, (detailed_y % 10) * small_block_size,
                        ((detailed_x % 10) + 1) * small_block_size, ((detailed_y % 10) + 1) * small_block_size,
                        outline='red', fill='', width=2
                    )
        elif self.marking_mode.get() == "background":
            for pixel_id in enclosed_blocks:
                detailed_x = pixel_id[0]
                detailed_y = pixel_id[1]
                block_x = base_x + (detailed_x * detailed_block_size)
                block_y = base_y + (detailed_y * detailed_block_size)
                if self.is_block_background(detailed_x, detailed_y, detailed_block_size):
                    self.remove_block_pixels_from_background(detailed_x, detailed_y, detailed_block_size)
                    self.redraw_detailed_block(detailed_x, detailed_y, canvas, small_block_size)
                else:
                    self.add_block_pixels_to_background(detailed_x, detailed_y, detailed_block_size)
                    canvas.create_rectangle(
                        (detailed_x % 10) * small_block_size, (detailed_y % 10) * small_block_size,
                        ((detailed_x % 10) + 1) * small_block_size, ((detailed_y % 10) + 1) * small_block_size,
                        outline='black', fill='', width=2
                    )

        self.drawn_lines = []
        canvas.delete('blue_line')
        self.update_processed_image()


    def get_detailed_enclosed_blocks(self, points, block_id, small_block_size):
        enclosed_blocks = set()
        for i in range(0, small_block_size * 10, small_block_size):
            for j in range(0, small_block_size * 10, small_block_size):
                pixel_x = block_id[0] * (self.block_size // 10) + (i // small_block_size)
                pixel_y = block_id[1] * (self.block_size // 10) + (j // small_block_size)
                if self.is_point_in_polygon(i, j, points) or self.is_point_in_polygon(i + small_block_size - 1, j, points) or self.is_point_in_polygon(i, j + small_block_size - 1, points) or self.is_point_in_polygon(i + small_block_size - 1, j + small_block_size - 1, points):
                    enclosed_blocks.add((pixel_x, pixel_y))
        return enclosed_blocks

    def redraw_detailed_block(self, x, y, canvas, small_block_size):
        canvas.create_rectangle(
            x * small_block_size, y * small_block_size,
            (x + 1) * small_block_size, (y + 1) * small_block_size,
            outline='white', fill=''
        )

    def open_third_level_window(self, block_id, detailed_block_id):
        third_level_window = tk.Toplevel(self.root)
        third_level_window.title(f"Third Level View: Block {block_id}, Detailed Block {detailed_block_id}")
        third_level_canvas = tk.Canvas(third_level_window)
        third_level_canvas.pack(fill=tk.BOTH, expand=True)
        
        detailed_block_image = self.image.crop(
            (block_id[0] * self.block_size + detailed_block_id[0] * (self.block_size // 10),
            block_id[1] * self.block_size + detailed_block_id[1] * (self.block_size // 10),
            block_id[0] * self.block_size + (detailed_block_id[0] + 1) * (self.block_size // 10),
            block_id[1] * self.block_size + (detailed_block_id[1] + 1) * (self.block_size // 10))
        )

        third_level_image = detailed_block_image.resize((self.block_size * 10, self.block_size * 10), Image.LANCZOS)
        third_level_photo = ImageTk.PhotoImage(third_level_image)
        
        third_level_canvas.config(width=third_level_photo.width(), height=third_level_photo.height())
        third_level_canvas.create_image(0, 0, image=third_level_photo, anchor=tk.NW)
        
        # Draw the smaller grid and mark foreground/background pixels
        pixel_size = third_level_photo.width() // 10
        for i in range(0, third_level_photo.width(), pixel_size):
            for j in range(0, third_level_photo.height(), pixel_size):
                pixel_id = (
                    block_id[0] * self.block_size + detailed_block_id[0] * (self.block_size // 10) + (i // pixel_size),
                    block_id[1] * self.block_size + detailed_block_id[1] * (self.block_size // 10) + (j // pixel_size)
                )
                if pixel_id in self.foreground_pixels:
                    third_level_canvas.create_rectangle(
                        i, j,
                        i + pixel_size, j + pixel_size,
                        outline='red', fill='', width=2
                    )
                elif pixel_id in self.background_pixels:
                    third_level_canvas.create_rectangle(
                        i, j,
                        i + pixel_size, j + pixel_size,
                        outline='black', fill='', width=2
                    )
                else:
                    third_level_canvas.create_line([(i, j), (i, j + pixel_size)], fill='white')
                    third_level_canvas.create_line([(i, j), (i + pixel_size, j)], fill='white')
                    third_level_canvas.create_line([(i + pixel_size, j), (i + pixel_size, j + pixel_size)], fill='white')
                    third_level_canvas.create_line([(i, j + pixel_size), (i + pixel_size, j + pixel_size)], fill='white')

        third_level_canvas.bind("<Button-1>", lambda event: self.start_third_level_draw_or_click(event, third_level_canvas, block_id, detailed_block_id, pixel_size))
        third_level_canvas.bind("<B1-Motion>", lambda event: self.draw_third_level(event, third_level_canvas, block_id, detailed_block_id, pixel_size))
        third_level_canvas.bind("<ButtonRelease-1>", lambda event: self.end_third_level_draw_or_click(event, third_level_canvas, block_id, detailed_block_id, pixel_size))

        # Bind right-click to mark_similar_color
        third_level_canvas.bind("<Button-2>", lambda event: self.mark_similar_color(event, third_level_canvas, block_id, detailed_block_id, pixel_size))

        # Store the image reference to avoid garbage collection
        third_level_canvas.image = third_level_photo


    def start_third_level_draw_or_click(self, event, canvas, block_id, detailed_block_id, pixel_size):
        self.drawing = False
        self.single_click_position = (event.x, event.y)
        self.moved = False
        self.drawn_lines = [[event.x, event.y]]


    def draw_third_level(self, event, canvas, block_id, detailed_block_id, pixel_size):
        self.moved = True
        if not self.drawing:
            self.drawing = True
            self.drawn_lines = [[event.x, event.y]]

        self.drawn_lines.append([event.x, event.y])
        canvas.create_line(
            self.drawn_lines[-2][0], self.drawn_lines[-2][1],
            self.drawn_lines[-1][0], self.drawn_lines[-1][1],
            fill='blue',
            tags='blue_line'
        )

        if self.marking_style.get() == "dragged":
            self.process_dragged_line_third_level(event.x, event.y, canvas, block_id, detailed_block_id, pixel_size)
            
    def process_dragged_line_third_level(self, x, y, canvas, block_id, detailed_block_id, pixel_size):
        if len(self.drawn_lines) < 2:
            return

        x1, y1 = self.drawn_lines[-2]
        x2, y2 = self.drawn_lines[-1]

        # Interpolate points between (x1, y1) and (x2, y2)
        num_points = max(abs(x2 - x1), abs(y2 - y1))  # Number of points to interpolate
        num_points = max(num_points, 1)  # Ensure num_points is at least 1 to avoid division by zero

        third_level_block_size = self.block_size // 100

        for i in range(num_points + 1):
            xi = x1 + i * (x2 - x1) / num_points
            yi = y1 + i * (y2 - y1) / num_points
            block_x = block_id[0] * self.block_size + detailed_block_id[0] * (self.block_size // 10) + int(xi // pixel_size)
            block_y = block_id[1] * self.block_size + detailed_block_id[1] * (self.block_size // 10) + int(yi // pixel_size)

            grid_x = (block_x % (self.block_size // 10)) // third_level_block_size
            grid_y = (block_y % (self.block_size // 10)) // third_level_block_size

            if self.marking_mode.get() == "foreground":
                if not self.is_block_foreground(block_x, block_y, third_level_block_size):
                    self.add_block_pixels_to_foreground(block_x, block_y, third_level_block_size)
                    canvas.create_rectangle(
                        grid_x * pixel_size,
                        grid_y * pixel_size,
                        (grid_x + 1) * pixel_size,
                        (grid_y + 1) * pixel_size,
                        outline='red', fill='', width=2
                    )
            elif self.marking_mode.get() == "background":
                if not self.is_block_background(block_x, block_y, third_level_block_size):
                    self.add_block_pixels_to_background(block_x, block_y, third_level_block_size)
                    canvas.create_rectangle(
                        grid_x * pixel_size,
                        grid_y * pixel_size,
                        (grid_x + 1) * pixel_size,
                        (grid_y + 1) * pixel_size,
                        outline='black', fill='', width=2
                    )

    def end_third_level_draw_or_click(self, event, canvas, block_id, detailed_block_id, pixel_size):
        if self.moved and self.drawing:
            self.drawn_lines.append([event.x, event.y])
            if self.marking_style.get() == "circled":
                self.process_third_level_drawn_area(canvas, block_id, detailed_block_id, pixel_size)
            self.drawing = False
            self.single_click_position = None
            canvas.delete('blue_line')
        else:
            self.process_third_level_single_click(event.x, event.y, canvas, block_id, detailed_block_id, pixel_size)
        self.save_state()
        self.update_processed_image()



    def process_third_level_single_click(self, x, y, canvas, block_id, detailed_block_id, pixel_size):
        third_level_x = block_id[0] * self.block_size + detailed_block_id[0] * (self.block_size // 10) + (x // pixel_size)
        third_level_y = block_id[1] * self.block_size + detailed_block_id[1] * (self.block_size // 10) + (y // pixel_size)
        third_level_block_size = self.block_size // 100

        if self.marking_mode.get() == "foreground":
            if self.is_block_foreground(third_level_x, third_level_y, third_level_block_size):
                self.remove_block_pixels_from_foreground(third_level_x, third_level_y, third_level_block_size)
                self.redraw_third_level_block((third_level_x % (self.block_size // 10)) // third_level_block_size,
                                            (third_level_y % (self.block_size // 10)) // third_level_block_size,
                                            canvas, pixel_size)
            else:
                self.add_block_pixels_to_foreground(third_level_x, third_level_y, third_level_block_size)
                canvas.create_rectangle(
                    (third_level_x % (self.block_size // 10)) // third_level_block_size * pixel_size,
                    (third_level_y % (self.block_size // 10)) // third_level_block_size * pixel_size,
                    ((third_level_x % (self.block_size // 10)) // third_level_block_size + 1) * pixel_size,
                    ((third_level_y % (self.block_size // 10)) // third_level_block_size + 1) * pixel_size,
                    outline='red', fill='', width=2
                )
        elif self.marking_mode.get() == "background":
            if self.is_block_background(third_level_x, third_level_y, third_level_block_size):
                self.remove_block_pixels_from_background(third_level_x, third_level_y, third_level_block_size)
                self.redraw_third_level_block((third_level_x % (self.block_size // 10)) // third_level_block_size,
                                            (third_level_y % (self.block_size // 10)) // third_level_block_size,
                                            canvas, pixel_size)
            else:
                self.add_block_pixels_to_background(third_level_x, third_level_y, third_level_block_size)
                canvas.create_rectangle(
                    (third_level_x % (self.block_size // 10)) // third_level_block_size * pixel_size,
                    (third_level_y % (self.block_size // 10)) // third_level_block_size * pixel_size,
                    ((third_level_x % (self.block_size // 10)) // third_level_block_size + 1) * pixel_size,
                    ((third_level_y % (self.block_size // 10)) // third_level_block_size + 1) * pixel_size,
                    outline='black', fill='', width=2
                )        
        self.update_processed_image()



    def process_third_level_drawn_area(self, canvas, block_id, detailed_block_id, pixel_size):
        if len(self.drawn_lines) < 3:
            canvas.delete('blue_line')
            self.drawn_lines = []
            return

        enclosed_blocks = self.get_third_level_enclosed_blocks(self.drawn_lines, block_id, detailed_block_id, pixel_size)
        third_level_block_size = self.block_size // 100

        if self.marking_mode.get() == "foreground":
            for pixel_id in enclosed_blocks:
                third_level_x = pixel_id[0]
                third_level_y = pixel_id[1]
                if self.is_block_foreground(third_level_x, third_level_y, third_level_block_size):
                    self.remove_block_pixels_from_foreground(third_level_x, third_level_y, third_level_block_size)
                    self.redraw_third_level_block((third_level_x % (self.block_size // 10)) // third_level_block_size,
                                                (third_level_y % (self.block_size // 10)) // third_level_block_size,
                                                canvas, pixel_size)
                else:
                    self.add_block_pixels_to_foreground(third_level_x, third_level_y, third_level_block_size)
                    canvas.create_rectangle(
                        (third_level_x % (self.block_size // 10)) // third_level_block_size * pixel_size,
                        (third_level_y % (self.block_size // 10)) // third_level_block_size * pixel_size,
                        ((third_level_x % (self.block_size // 10)) // third_level_block_size + 1) * pixel_size,
                        ((third_level_y % (self.block_size // 10)) // third_level_block_size + 1) * pixel_size,
                        outline='red', fill='', width=2
                    )
        elif self.marking_mode.get() == "background":
            for pixel_id in enclosed_blocks:
                third_level_x = pixel_id[0]
                third_level_y = pixel_id[1]
                if self.is_block_background(third_level_x, third_level_y, third_level_block_size):
                    self.remove_block_pixels_from_background(third_level_x, third_level_y, third_level_block_size)
                    self.redraw_third_level_block((third_level_x % (self.block_size // 10)) // third_level_block_size,
                                                (third_level_y % (self.block_size // 10)) // third_level_block_size,
                                                canvas, pixel_size)
                else:
                    self.add_block_pixels_to_background(third_level_x, third_level_y, third_level_block_size)
                    canvas.create_rectangle(
                        (third_level_x % (self.block_size // 10)) // third_level_block_size * pixel_size,
                        (third_level_y % (self.block_size // 10)) // third_level_block_size * pixel_size,
                        ((third_level_x % (self.block_size // 10)) // third_level_block_size + 1) * pixel_size,
                        ((third_level_y % (self.block_size // 10)) // third_level_block_size + 1) * pixel_size,
                        outline='black', fill='', width=2
                    )

        self.drawn_lines = []
        canvas.delete('blue_line')
        self.update_processed_image()

    def get_third_level_enclosed_blocks(self, points, block_id, detailed_block_id, pixel_size):
        enclosed_blocks = set()
        for i in range(0, pixel_size * 10, pixel_size):
            for j in range(0, pixel_size * 10, pixel_size):
                pixel_x = block_id[0] * self.block_size + detailed_block_id[0] * (self.block_size // 10) + (i // pixel_size)
                pixel_y = block_id[1] * self.block_size + detailed_block_id[1] * (self.block_size // 10) + (j // pixel_size)
                if self.is_point_in_polygon(i, j, points) or self.is_point_in_polygon(i + pixel_size - 1, j, points) or self.is_point_in_polygon(i, j + pixel_size - 1, points) or self.is_point_in_polygon(i + pixel_size - 1, j + pixel_size - 1, points):
                    enclosed_blocks.add((pixel_x, pixel_y))
        return enclosed_blocks

    def redraw_third_level_block(self, x, y, canvas, pixel_size):
        canvas.create_rectangle(
            x * pixel_size, y * pixel_size,
            (x + 1) * pixel_size, (y + 1) * pixel_size,
            outline='white', fill=''
        )   

    def mark_similar_color(self, event, canvas, block_id, detailed_block_id, pixel_size):
        x = event.x
        y = event.y
        third_level_x = block_id[0] * self.block_size + detailed_block_id[0] * (self.block_size // 10) + (x // pixel_size)
        third_level_y = block_id[1] * self.block_size + detailed_block_id[1] * (self.block_size // 10) + (y // pixel_size)
        third_level_block_size = self.block_size // 100

        target_color = self.image.getpixel((third_level_x, third_level_y))
        tolerance = self.tolerance_slider.get()  # Get tolerance from slider

        if self.marking_scope.get() == "local":
            for i in range(0, pixel_size * 10, pixel_size):
                for j in range(0, pixel_size * 10, pixel_size):
                    pixel_x = block_id[0] * self.block_size + detailed_block_id[0] * (self.block_size // 10) + (i // pixel_size)
                    pixel_y = block_id[1] * self.block_size + detailed_block_id[1] * (self.block_size // 10) + (j // pixel_size)

                    current_color = self.image.getpixel((pixel_x, pixel_y))

                    if all(abs(current_color[k] - target_color[k]) < tolerance for k in range(3)):
                        if self.marking_mode.get() == "foreground":
                            self.add_block_pixels_to_foreground(pixel_x, pixel_y, third_level_block_size)
                            canvas.create_rectangle(
                                (pixel_x % (self.block_size // 10)) // third_level_block_size * pixel_size,
                                (pixel_y % (self.block_size // 10)) // third_level_block_size * pixel_size,
                                ((pixel_x % (self.block_size // 10)) // third_level_block_size + 1) * pixel_size,
                                ((pixel_y % (self.block_size // 10)) // third_level_block_size + 1) * pixel_size,
                                outline='red', fill='', width=2
                            )
                        elif self.marking_mode.get() == "background":
                            self.add_block_pixels_to_background(pixel_x, pixel_y, third_level_block_size)
                            canvas.create_rectangle(
                                (pixel_x % (self.block_size // 10)) // third_level_block_size * pixel_size,
                                (pixel_y % (self.block_size // 10)) // third_level_block_size * pixel_size,
                                ((pixel_x % (self.block_size // 10)) // third_level_block_size + 1) * pixel_size,
                                ((pixel_y % (self.block_size // 10)) // third_level_block_size + 1) * pixel_size,
                                outline='black', fill='', width=2
                            )
            self.save_state()
            self.update_processed_image()
        elif self.marking_scope.get() == "global":
            if not self.modification_mode:
                self.enter_modification_mode()

            self.temp_foreground_pixels = set()
            self.temp_background_pixels = set()

            self.reference_color = target_color
            self.reference_block_id = block_id
            self.reference_detailed_block_id = detailed_block_id
            self.reference_pixel_size = pixel_size

            self.update_display_in_modification_mode(None)

    def redraw_block(self, x, y):
        # self.canvas.create_rectangle(
        #     block_x * self.block_size, block_y * self.block_size,
        #     (block_x + 1) * self.block_size, (block_y + 1) * self.block_size,
        #     outline='white', fill=''
        # )
        pass

    def update_processed_image(self):
        draw = ImageDraw.Draw(self.processed_image)
        for i in range(0, self.processed_image.width):
            for j in range(0, self.processed_image.height):
                if (i, j) in self.foreground_pixels:
                    draw.point((i, j), fill="white")
                elif (i, j) in self.background_pixels:
                    draw.point((i, j), fill="black")
                else:
                    draw.point((i, j), fill = "grey")
        self.processed_photo = ImageTk.PhotoImage(self.processed_image)
        self.processed_canvas.create_image(0, 0, image=self.processed_photo, anchor=tk.NW)
        self.update_display()

    def save_labels(self):
        if not self.image_path:
            return
        label_path = self.image_path.rsplit('.', 1)[0] + '_labels.json'
        labels = {
            "foreground": list(self.foreground_pixels),
            "background": list(self.background_pixels)
        }
        with open(label_path, 'w') as f:
            json.dump(labels, f)
        print(f"Labels saved to {label_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabeler(root)
    root.mainloop()
