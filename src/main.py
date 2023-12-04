from fastapi import FastAPI, Form
import numpy as np
import pytesseract
import cv2
import requests
import uvicorn
import json

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the OCR API"}


@app.post("/ocr-from-url/")
async def perform_ocr_from_url(url: str = Form(...)):
    response = requests.get(url)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    config = r"-l eng --oem 3 --psm 6"
    text = pytesseract.image_to_string(thresh, config=config)

    data = []
    for row in text.split("\n"):
        if row == "":
            continue

        row = row.replace("M", "")
        row_elements = []
        for idx, value in enumerate(row.split(" ")):
            if idx == 0:
                value += "M"

            if value:
                row_elements.append(value)

        print(f"{row_elements}")
        data.append(row_elements)

    return data