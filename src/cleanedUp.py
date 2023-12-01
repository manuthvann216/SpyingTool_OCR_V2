import cv2
import numpy as np
from io import BytesIO
import requests
import pandas as pd
import json
import easyocr

# Define the EasyOCR Reader
reader = easyocr.Reader(['en'])

def read_image_from_url(url):
    response = requests.get(url)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def perform_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use EasyOCR to extract text
    results = reader.readtext(gray)

    # Extracting text without coordinates
    text_data = [result[1] for result in results]

    # Convert list to JSON
    json_data = json.dumps(text_data)

    return json_data


# Example usage
url = "https://i.imgur.com/oU0Sxfy.png"
image = read_image_from_url(url)
result = perform_ocr(image)
print(result)
