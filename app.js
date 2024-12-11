// app.js

const canvas = document.getElementById('mainCanvas');
const ctx = canvas.getContext('2d');

// Load image
const img = new Image();
img.src = '/path_to_image.jpg';  // Ensure the backend serves this
img.onload = () => {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
};

// Handle marking
document.getElementById('markForeground').addEventListener('click', () => {
    // Capture coordinates (e.g., from mouse events)
    const coordinates = []; // Populate with actual coordinates
    fetch('/api/mark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ coordinates, mode: 'foreground' })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // Update the canvas or fetch the processed image
            refreshProcessedImage();
        }
    });
});

document.getElementById('markBackground').addEventListener('click', () => {
    const coordinates = []; // Populate with actual coordinates
    fetch('/api/mark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ coordinates, mode: 'background' })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            refreshProcessedImage();
        }
    });
});

function refreshProcessedImage() {
    const processedImg = new Image();
    processedImg.src = '/api/processed_image';
    processedImg.onload = () => {
        ctx.drawImage(processedImg, 0, 0, canvas.width, canvas.height);
    };
}
