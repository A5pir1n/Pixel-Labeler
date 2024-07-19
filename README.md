```markdown
# My Image Labeler

A tool for labeling foreground and background in images with advanced features such as local/global mode, customizable tolerance, redo/undo functionality, and integration with image recognition models.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/my-image-labeler.git
   cd my-image-labeler
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the labeling script:
```bash
python labeler.py
```

## Features

### Main Window

- **Load Image**: Load an image from your local system to start labeling.
- **Save Labels**: Save the labeled data to a file.
- **Load Labels**: Load previously saved labels to continue working on them.
- **Clear All**: Clear all labels and reset the image.
- **Classify by SAM**: Use Meta's Segment Anything Model (SAM) to classify the image based on thresholds set by the user.
- **Load Custom Model**: Load a custom image recognition model for classification.
- **Redo**: Redo the last undone action.
- **Undo**: Undo the last action.
- **Lambda Slider**: Adjust the display between the original image and the labeled black/white image.
- **Tolerance Slider**: Adjust the tolerance level for color similarity when marking similar colors.
- **Marking Foreground/Background**: Toggle between marking foreground and background.
- **Marking on Circled/Dragged Areas**: Choose between marking enclosed areas or dragged lines.
- **Local/Global Mode**: Toggle between local and global marking modes.

### Detailed Windows

Right-clicking on a grid in the main window opens a detailed view. Right-clicking on a grid in the detailed window opens a more detailed view. The following functionalities are available:

- **Single Click**: Toggle the grid's status between foreground, background, and unidentified.
- **Hold and Drag**: Draw lines to mark multiple grids.
- **Right Click (Local Mode)**: Mark all similar colors within the current detailed window.
- **Right Click (Global Mode)**: Enter modification mode to mark similar colors globally with adjustable tolerance.

### Modification Mode (Global Only)

When marking similar colors globally:
- **Apply**: Apply the changes.
- **Cancel**: Discard the changes.
- **Adjust Tolerance**: Dynamically adjust the tolerance level during modification mode.

## Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository.
2. Create a new branch with your feature or bugfix.
3. Submit a pull request.

### Uploading Your Labeled Data

Save your labeled image and metadata in the `user_uploads/` directory. Make sure to include:
- The labeled image.
- The label file (e.g., `image_labels.txt`).
- A prompt file (`prompt.txt`) with a short description of what you labeled (e.g., marking foreground for the cherry tree).

Submit a pull request with your uploads.

## License

This project is licensed under the MIT License.
