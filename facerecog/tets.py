import dlib
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Load image from URL
url = "https://upload.wikimedia.org/wikipedia/commons/c/c1/Shah_Rukh_Khan_in_2023_%281%29.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB")
img_np = np.ascontiguousarray(np.array(img))

print("Shape:", img_np.shape, "Dtype:", img_np.dtype)

detector = dlib.get_frontal_face_detector()
faces = detector(img_np, 1)  # 1 = upsample once

print(f"Found {len(faces)} face(s)")
