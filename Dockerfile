# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /spyingtool_ocr

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y tesseract-ocr poppler-utils ffmpeg libsm6 libxext6

# Copy the main.py file and other necessary files into the container at /app
COPY ./src/main.py ./requirements.txt /spyingtool_ocr/

RUN pip3 install --no-cache-dir -r requirements.txt


EXPOSE 80

# Set environment to production
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV ENVIRONMENT production

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
