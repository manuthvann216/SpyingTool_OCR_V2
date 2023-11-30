from fastapi import FastAPI, Form
import cv2
import numpy as np
import pytesseract
from io import BytesIO
import requests

app = FastAPI()


def read_image_from_url(url):
    response = requests.get(url)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def perform_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh)
    return text


@app.post("/ocr-from-url/")
async def perform_ocr_from_url(url: str = Form(...)):
    image = read_image_from_url(url)
    text = perform_ocr(image)
    return {"url": url, "text": text}
