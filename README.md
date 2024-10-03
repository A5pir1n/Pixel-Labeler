# Pixel Labeler

A powerful tool for labeling images with foreground and background segmentation, featuring advanced functionalities such as local and global marking modes, customizable tolerance levels, and integration with image recognition models.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/A5pir1n/Pixel-Labeler.git
   cd Pixel-Labeler
   ```

2. **Install Required Dependencies:**

   ```bash
   pip3 install -r requirements.txt
   ```

---

## Usage

Run the labeling script:

```bash
python3 label.py
```

---

## Features

### Main Window

- **Load Image**: Load an image from your local system to start labeling.
- **Save Labels**: Save the labeled data to a folder containing both a `.npy` file and a black and white image of the labels.
- **Load Labels**: Load previously saved labels to continue working on them.
- **Clear All**: Clear all labels and reset the image.
- **Classify by SAM**: Use Meta's Segment Anything Model (SAM) to classify the image based on thresholds set by the user.
- **Load Custom Model**: Load a custom image recognition model for classification.
- **Undo/Redo**: Undo or redo actions to facilitate efficient labeling.
- **Lambda Slider**: Adjust the display blend between the original image and the labeled image.
- **Tolerance Slider**: Adjust the tolerance level for color similarity when marking similar colors.
- **Marking Modes**:
  - **Marking Foreground**: Mark selected areas as foreground.
  - **Marking Background**: Mark selected areas as background.
  - **Mark Unidentified**: Reset selected areas to unidentified.
- **Marking Styles**:
  - **Circled Areas**: Enclose areas to mark them.
  - **Dragged Lines**: Draw lines to mark areas.
- **Local/Global Mode**: Toggle between local and global marking modes.
- **Lock/Unlock Pixels**: Lock the current marked pixels to prevent further modifications, and unlock them when needed.

---

### Detailed Windows

#### Accessing Detailed Views

- **Right-Click on Main Grid**: Opens a detailed view (second-level window) of the selected grid.
- **Right-Click on Detailed Grid**: Opens a more detailed view (third-level window) of the selected grid in the detailed window.

#### Functionalities in Detailed Views

- **Single Click**:
  - **Toggle Marking**: Click on a grid to toggle its status between foreground, background, and unidentified based on the current marking mode.
- **Hold and Drag**:
  - **Mark Multiple Grids**: Draw lines to mark multiple grids quickly.
- **Shift + Left Click**:
  - **Mark Similar Colors Locally**: In the detailed window, hold the Shift key and left-click on a grid to mark all similar colors within the detailed window based on average color and tolerance level.
- **Right Click (Global Mode)**:
  - **Enter Modification Mode**: Mark similar colors globally across the entire image.
  
---

### Modification Mode (Global Only)

When marking similar colors globally:

- **Apply**: Apply the changes made during modification mode.
- **Cancel**: Discard the changes and revert to the previous state.
- **Adjust Tolerance**: Dynamically adjust the tolerance level during modification mode to fine-tune the selection.

---

### Lock and Unlock Pixels

- **Lock**: Lock the current marked pixels (both foreground and background) to prevent further modifications. Only unidentified pixels can be modified after locking.
- **Unlock**: Unlock all pixels, allowing them to be modified again.

---

## Contributing

We welcome contributions! Here are three ways you can help:

### 1. Uploading and Labeling New Images

Upload a new image, attach the labels, and provide a description.

**Steps:**

1. **Load an Image**: Click "Load Image" to upload your image.
2. **Label the Image**: Use the tool to label the image.
3. **Save Labels**: Click "Save Labels" when finished labeling.
4. **Prepare Folder**:
   - Create a folder with the image name as the title.
   - Include the original image.
   - Include the labels folder generated (contains `.npy` file and label image).
   - Add a `description.txt` file with a one-sentence description of the foreground object.
5. **Submit a Pull Request**: Upload the folder to the repository and submit a pull request.

### 2. Labeling Existing Unlabeled Images

Select an unlabeled image and start labeling.

**Steps:**

1. **Select an Unlabeled Image**: Choose an image from the "Unlabeled" folder.
2. **Label the Image**: Use the tool to label the image.
3. **Save Labels**: Click "Save Labels" when finished labeling.
4. **Update Folder**:
   - Go to the same folder.
   - Include the labels folder generated.
   - Add a `description.txt` file with a one-sentence description of the foreground object.
5. **Submit a Pull Request**: Upload the updated folder to the repository and submit a pull request.

### 3. Challenging Existing Labeling

Redo the labels for an existing image to improve or correct them.

**Steps:**

1. **Load an Image with Existing Labels**: Use "Load Image" and "Load Labels" to open an image and its labels.
2. **Clear or Modify Labels**: Use "Clear All" if necessary, or modify the existing labels.
3. **Re-label the Image**: Use the tool to re-label the image.
4. **Save Labels**: Click "Save Labels" when finished labeling.
5. **Update Folder**:
   - Go to the same folder.
   - Replace or update the labels folder.
   - Update the `description.txt` if necessary.
6. **Submit a Pull Request**: Upload the updated folder to the repository and submit a pull request.

---

## License

This project is licensed under the MIT License.

---
