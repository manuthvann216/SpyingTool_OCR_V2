from fastapi import FastAPI, Form
import numpy as np
import pytesseract
import cv2
import requests
import uvicorn
import json 
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

    return tabular_data

def organize_data(data):
    organized_data = {"data": [[]]}
    current_row = 0

    for i, value in enumerate(data):
        if i % 7 == 0 and i != 0:
            organized_data["data"].append([])
            current_row += 1

        organized_data["data"][current_row].append(value)

    return organized_data

def convert_to_specific_format(organized_data):
    formatted_data = {"data": [[]]}
    
    for row in organized_data["data"]:
        formatted_row = []
        for i in range(len(row[0])):
            formatted_cell = [
                row[j][i] if i < len(row[j]) else "" for j in range(len(row))
            ]
            formatted_row.append(formatted_cell)

        formatted_data["data"].append(formatted_row)

    return formatted_data

def additional_step(specific_text):
    transformed_data = {"data": [[]]}

    for row in specific_text["data"][1:]:
        transformed_row = []
        for item in row:
            new_item = [
                item[0].replace(" ", ""),
                item[2],
                item[4],
                item[5],
                item[1].replace("M", ""),
                item[3],
                item[6],
                item[1].replace("M", "") if item[1] != "24-36 M" else "",
            ]
            transformed_row.append(new_item)

        transformed_data["data"][0].extend(transformed_row)

    return transformed_data

def final_step(half_way_done_data):

    formatted_data = []

    for sublist in half_way_done_data["data"][0]:
        formatted_sublist = [
            sublist[0].replace("M", " M").replace(" ", ""),  # Format the first element
            sublist[1],
            sublist[2],
            sublist[5],
            sublist[4].replace(" ", "") + " " + sublist[1],
            sublist[7] if sublist[7] != "" else "0.00%",  # Handle the case where the last element is an empty string
            sublist[4].replace(" ", "") + " " + sublist[2],
            sublist[4].replace(" ", "") + " " + sublist[6]
        ]
        formatted_data.append(formatted_sublist)

    output_data = {"data": [formatted_data]}

    return output_data

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

    # Organize the data into the desired structure
    organized_data = organize_data(tabular_data)

    # Convert to the specific format
    formatted_data = convert_to_specific_format(organized_data)

    #additional  step 
    # half_way_done_result = additional_step(formatted_data) 

    #final step
    # result = final_step(half_way_done_result)

    return formatted_data
