# app.py

from flask import Flask, request, jsonify, send_file
from image_processing import ImageProcessor

app = Flask(__name__)

# Initialize ImageProcessor (you might want to handle multiple sessions/users)
processor = ImageProcessor("path_to_image.jpg")

@app.route('/api/mark', methods=['POST'])
def mark():
    data = request.json
    coordinates = data.get('coordinates')
    mode = data.get('mode')  # 'foreground' or 'background'
    
    if mode == 'foreground':
        processor.mark_foreground(coordinates)
    elif mode == 'background':
        processor.mark_background(coordinates)
    else:
        return jsonify({'error': 'Invalid mode'}), 400
    
    return jsonify({'status': 'success'})

@app.route('/api/processed_image', methods=['GET'])
def get_processed_image():
    processor.get_processed_image()
    return send_file("processed_image.png", mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
