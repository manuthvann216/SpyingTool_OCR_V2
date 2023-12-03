from fastapi import FastAPI, Form
import numpy as np
import pytesseract
import cv2
from io import BytesIO
import requests
import matplotlib.pyplot as plt

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

def extract_tabular_data(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin_otsu = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[1]//100))
    eroded_image = cv2.erode(img_bin_otsu, vertical_kernel, iterations=3)
    vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=3)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1]//100, 1))
    horizontal_lines = cv2.erode(img_bin_otsu, hor_kernel, iterations=5)
    horizontal_lines = cv2.dilate(horizontal_lines, hor_kernel, iterations=5)
    vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, vertical_kernel, iterations=3)
    _, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bitxor = cv2.bitwise_xor(img, vertical_horizontal_lines)
    bitnot = cv2.bitwise_not(bitxor)

    contours, hierarchy = cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda x:x[1][1]))

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w < 1000 and h < 500):
            image = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            boxes.append([x, y, w, h])

    rows = []
    columns = []
    heights = [boxes[i][3] for i in range(len(boxes))]
    mean = np.mean(heights)
    columns.append(boxes[0])
    previous = boxes[0]

    for i in range(1, len(boxes)):
        if(boxes[i][1] <= previous[1] + mean/2):
            columns.append(boxes[i])
            previous = boxes[i]
            if(i == len(boxes)-1):
                rows.append(columns)
        else:
            rows.append(columns)
            columns = []
            previous = boxes[i]
            columns.append(boxes[i])

    total_cells = 0
    for i in range(len(rows)):
        if len(rows[i]) > total_cells:
            total_cells = len(rows[i])

    center = [int(rows[i][j][0]+rows[i][j][2]/2) for j in range(len(rows[i])) if rows[0]]
    center = np.array(center)
    center.sort()

    boxes_list = []
    for i in range(len(rows)):
        l = []
        for k in range(total_cells):
            l.append([])
        for j in range(len(rows[i])):
            diff = abs(center - (rows[i][j][0] + rows[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            l[indexing].append(rows[i][j])
        boxes_list.append(l)

    dataframe_final = []
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            s = ''
            if(len(boxes_list[i][j]) == 0):
                dataframe_final.append(' ')
            else:
                for k in range(len(boxes_list[i][j])):
                    y, x, w, h = boxes_list[i][j][k][0], boxes_list[i][j][k][1], boxes_list[i][j][k][2], boxes_list[i][j][k][3]
                    roi = bitnot[x:x+h, y:y+w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel, iterations=1)
                    erosion = cv2.erode(dilation, kernel, iterations=2)                
                    out = pytesseract.image_to_string(erosion)
                    if(len(out) == 0):
                        out = pytesseract.image_to_string(erosion)
                    s = s + " " + out
                dataframe_final.append(s)

    arr = np.array(dataframe_final)
    import pandas as pd
    dataframe = pd.DataFrame(arr.reshape(len(rows), total_cells))
    return {"data": [dataframe.values.transpose().tolist()]}

@app.get("/")
def read_root():
    return {"message": "Welcome to the OCR API"}

@app.post("/ocr-from-url/")
async def perform_ocr_from_url(url: str = Form(...)):
    response = requests.get(url)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Extract tabular data
    tabular_data = extract_tabular_data(image)

    return tabular_data
