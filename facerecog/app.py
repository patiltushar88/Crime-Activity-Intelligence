from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import face_recognition
from PIL import Image, ImageFile
import requests
from io import BytesIO

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
CORS(app)

# ✅ Make image_database dynamic (initially empty)
image_database = {}

def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_np = np.array(img)
        img_np = np.ascontiguousarray(img_np)

        print(f"[LOAD] Image shape: {img_np.shape}, dtype: {img_np.dtype}, contiguous: {img_np.flags['C_CONTIGUOUS']}")
        return img_np
    except Exception as e:
        raise ValueError(f"Failed to load/process image from {url}: {e}")

def match_faces(image_url):
    try:
        unknown_image = load_image_from_url(image_url)
        print("[ENCODE] Attempting to encode uploaded image...")
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        if not unknown_encodings:
            raise ValueError("No face found in the input image.")

        unknown_encoding = unknown_encodings[0]
    except Exception as e:
        print(f"[ERROR] During encoding: {e}")
        return None, str(e)

    matches = {}
    for name, known_url in image_database.items():
        try:
            known_image = load_image_from_url(known_url)
            known_encodings = face_recognition.face_encodings(known_image)

            if not known_encodings:
                raise ValueError(f"No face found in database image for {name}.")

            known_encoding = known_encodings[0]
            result = face_recognition.compare_faces([known_encoding], unknown_encoding)
            matches[name] = str(result[0])
        except Exception as e:
            print(f"[WARNING] Error processing {name}: {e}")
            matches[name] = "False"

    return matches, None

@app.route('/match_faces', methods=['POST'])
def match_faces_api():
    data = request.get_json()

    # ✅ Accept image_database from frontend
    global image_database
    if 'image_database' in data:
        image_database = data['image_database']

    if 'image_url' not in data:
        return jsonify({'error': 'Image URL is required'}), 400

    image_url = data['image_url']
    matches, error = match_faces(image_url)

    if matches is None:
        return jsonify({'error': error}), 500

    return jsonify(matches), 200

if __name__ == '__main__':
    app.run(debug=True)
