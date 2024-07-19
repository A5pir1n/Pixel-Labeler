# My Image Labeler

A tool for labeling foreground and background in images with advanced features such as local/global mode, customizable tolerance, redo/undo functionality, and integration with image recognition models.


## Installation

1. Fork the repository on GitHub:
   - Go to the repository page on GitHub.
   - Click the "Fork" button in the upper right corner.
   - Clone your forked repository:
     ```bash
     git clone https://github.com/yourusername/Pixel-Labeler.git
     cd Pixel-Labeler.git
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

We welcome contributions! Here are three ways you can help:
### 1. Uploading and Labeling New Images

Upload a new image, attach a `label.json` file with existing labels, and provide a description. Follow these steps:
1. Click "Load Image" to upload your image.
2. Use the tool to label the image.
3. Click "Save Labels" when finished labelling.
5. Create a folder with image name as the title
6. Upload the original picture, label.json, and description.txt that uses one sentence to describe the foreground object
7. Submit a pull request.

### 2. Labeling Existing Unlabeled Images

Open an existing, unlabeled picture and start labeling:
1. Select an unlabeled image from the folder "Unlabeled".
2. Use the tool to label the image.
3. Click "Save Labels" when finished labelling.
5. Go to the same folder 
6. Upload the original picture, label.json, and description.txt that uses one sentence to describe the foreground object
7. Submit a pull request.


### 3. Challenging Existing Labeling

Challenge an existing labeling by redoing the labels for a certain photo:
1. Load an image with existing labels.
2. Clear the current labels if necessary.
3. Re-label the image using the tool.
3. Click "Save Labels" when finished labelling.
5. Go to the same folder 
6. Upload the original picture, label.json, and description.txt that uses one sentence to describe the foreground object
7. Submit a pull request.


## License

This project is licensed under the MIT License.
