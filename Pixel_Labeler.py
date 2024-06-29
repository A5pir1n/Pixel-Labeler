import tkinter as tk
import json
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageOps
import sys


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
        self.canvas = tk.Canvas(root)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.processed_canvas = tk.Canvas(root)
        self.processed_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.load_image_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_image_button.pack(side=tk.BOTTOM)
        self.save_labels_button = tk.Button(root, text="Save Labels", command=self.save_labels)
        self.save_labels_button.pack(side=tk.BOTTOM)

        self.clear_button = tk.Button(root, text="Clear All", command=self.clear_all)
        self.clear_button.pack(side=tk.BOTTOM)

        self.marking_mode = tk.StringVar(value="foreground")
        self.foreground_checkbox = tk.Radiobutton(root, text="Marking Foreground", variable=self.marking_mode, value="foreground")
        self.background_checkbox = tk.Radiobutton(root, text="Marking Background", variable=self.marking_mode, value="background")
        self.foreground_checkbox.pack(side=tk.TOP, anchor=tk.W)
        self.background_checkbox.pack(side=tk.TOP, anchor=tk.W)

        self.marking_style = tk.StringVar(value="circled")
        self.circled_area_checkbox = tk.Radiobutton(root, text="Marking on Circled Areas", variable=self.marking_style, value="circled")
        self.dragged_area_checkbox = tk.Radiobutton(root, text="Marking on Dragged Areas", variable=self.marking_style, value="dragged")
        self.circled_area_checkbox.pack(side=tk.TOP, anchor=tk.W)
        self.dragged_area_checkbox.pack(side=tk.TOP, anchor=tk.W)

        self.drawing = False
        self.drawn_lines = []
        self.single_click_position = None
        self.moved = False

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
            if self.marking_style.get() == "circled":
                self.process_drawn_area()
            self.canvas.delete('blue_line')
        else:
            self.process_single_click(event.x, event.y) 
        self.drawing = False
        self.single_click_position = None
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
                self.redraw_block(block_x, block_y)
            else:
                self.add_block_pixels_to_foreground(block_x, block_y)
                self.canvas.create_rectangle(
                    block_x * self.block_size, block_y * self.block_size,
                    (block_x + 1) * self.block_size, (block_y + 1) * self.block_size,
                    outline='red', fill='', width=2
                )

        elif self.marking_mode.get() == "background":
            if self.is_block_background(block_x, block_y):
                self.remove_block_pixels_from_background(block_x, block_y)
                self.redraw_block(block_x, block_y)
            else:
                self.add_block_pixels_to_background(block_x, block_y)
                self.canvas.create_rectangle(
                    block_x * self.block_size, block_y * self.block_size,
                    (block_x + 1) * self.block_size, (block_y + 1) * self.block_size,
                    outline='black', fill='', width=2
                )

        self.update_processed_image()

    def process_drawn_area(self):
        if len(self.drawn_lines) < 3:
            self.canvas.delete('blue_line')
            self.drawn_lines = []
            return

        enclosed_blocks = self.get_enclosed_blocks(self.drawn_lines)

        if self.marking_mode.get() == "foreground":
            for block_id in enclosed_blocks:
                if self.is_block_foreground(block_id[0], block_id[1]):
                    self.remove_block_pixels_from_foreground(block_id[0], block_id[1])
                    self.redraw_block(block_id[0], block_id[1])
                else:
                    self.add_block_pixels_to_foreground(block_id[0], block_id[1])
                    self.canvas.create_rectangle(
                        block_id[0] * self.block_size, block_id[1] * self.block_size,
                        (block_id[0] + 1) * self.block_size, (block_id[1] + 1) * self.block_size,
                        outline='red', fill='', width=2
                    )
        elif self.marking_mode.get() == "background":
            for block_id in enclosed_blocks:
                if self.is_block_background(block_id[0], block_id[1]):
                    self.remove_block_pixels_from_background(block_id[0], block_id[1])
                    self.redraw_block(block_id[0], block_id[1])
                else:
                    self.add_block_pixels_to_background(block_id[0], block_id[1])
                    self.canvas.create_rectangle(
                        block_id[0] * self.block_size, block_id[1] * self.block_size,
                        (block_id[0] + 1) * self.block_size, (block_id[1] + 1) * self.block_size,
                        outline='black', fill='', width=2
                    )

        self.drawn_lines = []
        self.canvas.delete('blue_line')
        self.update_processed_image()


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
        self.update_processed_image()



    def process_third_level_single_click(self, x, y, canvas, block_id, detailed_block_id, pixel_size):
        third_level_x = block_id[0] * (self.block_size // 10) + detailed_block_id[0] * (self.block_size // 100) + (x // pixel_size)
        third_level_y = block_id[1] * (self.block_size // 10) + detailed_block_id[1] * (self.block_size // 100) + (y // pixel_size)

        if self.marking_mode.get() == "foreground":
            if self.is_block_foreground(third_level_x, third_level_y, self.block_size // 100):
                self.remove_block_pixels_from_foreground(third_level_x, third_level_y, self.block_size // 100)
                self.redraw_third_level_block(x // pixel_size, y // pixel_size, canvas, pixel_size)
            else:
                self.add_block_pixels_to_foreground(third_level_x, third_level_y, self.block_size // 100)
                canvas.create_rectangle(
                    (x // pixel_size) * pixel_size, (y // pixel_size) * pixel_size,
                    ((x // pixel_size) + 1) * pixel_size, ((y // pixel_size) + 1) * pixel_size,
                    outline='red', fill='', width=2
                )
        elif self.marking_mode.get() == "background":
            if self.is_block_background(third_level_x, third_level_y, self.block_size // 100):
                self.remove_block_pixels_from_background(third_level_x, third_level_y, self.block_size // 100)
                self.redraw_third_level_block(x // pixel_size, y // pixel_size, canvas, pixel_size)
            else:
                self.add_block_pixels_to_background(third_level_x, third_level_y, self.block_size // 100)
                canvas.create_rectangle(
                    (x // pixel_size) * pixel_size, (y // pixel_size) * pixel_size,
                    ((x // pixel_size) + 1) * pixel_size, ((y // pixel_size) + 1) * pixel_size,
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

    def redraw_block(self, x, y):
        block_image = self.image.crop(
            (x * self.block_size, y * self.block_size, (x + 1) * self.block_size, (y + 1) * self.block_size)
        )
        block_photo = ImageTk.PhotoImage(block_image)
        self.canvas.create_image(
            x * self.block_size, y * self.block_size, image=block_photo, anchor=tk.NW
        )
        # Redraw the grid lines to maintain the white borders
        self.canvas.create_line(
            x * self.block_size, y * self.block_size, (x + 1) * self.block_size, y * self.block_size, fill='white', width=1
        )
        self.canvas.create_line(
            x * self.block_size, y * self.block_size, x * self.block_size, (y + 1) * self.block_size, fill='white', width=1
        )
        self.canvas.create_line(
            (x + 1) * self.block_size, y * self.block_size, (x + 1) * self.block_size, (y + 1) * self.block_size, fill='white', width=1
        )
        self.canvas.create_line(
            x * self.block_size, (y + 1) * self.block_size, (x + 1) * self.block_size, (y + 1) * self.block_size, fill='white', width=1
        )
        # Store the image reference to avoid garbage collection
        self.canvas.image = block_photo

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
