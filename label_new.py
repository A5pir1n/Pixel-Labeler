import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

def main():
    st.title("Image Labeler Web App")
    st.sidebar.title("Control Panel")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    else:
        st.warning("Please upload an image to begin.")
        st.stop()

    # Initialize labels in session state
    if 'labels' not in st.session_state:
        st.session_state['labels'] = np.full((image.height, image.width), 128, dtype=np.uint8)
    labels = st.session_state['labels']

    # Marking mode
    marking_mode = st.sidebar.radio(
        "Marking Mode",
        ('Foreground', 'Background', 'Unidentified'),
        index=0
    )

    # Drawing mode
    drawing_mode = st.sidebar.selectbox(
        "Drawing Tool",
        ("freedraw", "line", "rect", "circle", "transform")
    )

    stroke_width = st.sidebar.slider("Stroke Width", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke Color", "#FF0000")
    lambda_value = st.sidebar.slider("Lambda Value", 0.0, 1.0, 0.5)

    # Canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=image,
        update_streamlit=True,
        height=500,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # Update labels
    labels = update_labels_from_canvas(canvas_result, labels, marking_mode)

    # Create processed image
    processed_image = create_processed_image(labels)
    blended_image = Image.blend(image.convert('RGBA'), processed_image.convert('RGBA'), alpha=lambda_value)
    st.image(blended_image, caption='Blended Image', use_column_width=True)

    # Save labels
    if st.sidebar.button('Save Labels'):
        labels_image = Image.fromarray(labels)
        labels_image.save('labels.png')
        st.success('Labels saved.')

    # Load labels
    uploaded_labels = st.sidebar.file_uploader("Load Labels", type=["png", "npy"])
    if uploaded_labels is not None:
        if uploaded_labels.name.endswith('.png'):
            labels = np.array(Image.open(uploaded_labels))
        elif uploaded_labels.name.endswith('.npy'):
            labels = np.load(uploaded_labels)
        st.session_state['labels'] = labels
        st.success('Labels loaded.')

def update_labels_from_canvas(canvas_result, labels, marking_mode):
    if canvas_result.json_data is not None:
        for obj in canvas_result.json_data["objects"]:
            # Process each object (e.g., path, rect)
            pass  # Implement object processing
    return labels

def create_processed_image(labels):
    rgb_array = np.zeros((*labels.shape, 3), dtype=np.uint8)
    rgb_array[labels == 0] = [0, 0, 0]
    rgb_array[labels == 128] = [128, 128, 128]
    rgb_array[labels == 255] = [255, 255, 255]
    processed_image = Image.fromarray(rgb_array)
    return processed_image

if __name__ == "__main__":
    main()
