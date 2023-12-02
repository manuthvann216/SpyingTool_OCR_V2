from fastapi import FastAPI, Form
import numpy as np
import pytesseract
import cv2
from io import BytesIO
import requests

app = FastAPI()

def format_table_data(text):
    lines = text.strip().split('\n')

    num_columns = 12
    num_rows = len(lines) // num_columns

   
    tabular_data = []

    
    for i in range(num_rows):
        # Extract columns for the current row
        row_columns = lines[i * num_columns: (i + 1) * num_columns]

        # Create a list to store the columns for the current row
        row_data = []

        # Iterate over columns
        for column in row_columns:
            # Split each column into individual values
            column_values = column.strip().split()

            # If there are multiple values in a column, keep them together
            if len(column_values) > 1:
                row_data.extend([' '.join(column_values)])
            else:
                row_data.extend(column_values)

        tabular_data.append(row_data)

    return {"data": [tabular_data]}



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
    text = pytesseract.image_to_string(thresh)

    # Format the table data
    tabular_data = format_table_data(text)

    return {"data": tabular_data}
