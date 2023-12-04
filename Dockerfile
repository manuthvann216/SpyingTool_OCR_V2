# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment to production
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV ENVIRONMENT production

WORKDIR /spyingtool_ocr

RUN apt-get update -y && \
    apt-get install -y tesseract-ocr poppler-utils ffmpeg libsm6 libxext6

# Copy the main.py file and other necessary files into the container at /app
COPY ./requirements.txt /spyingtool_ocr/

RUN pip3 install -r requirements.txt

ADD . /spyingtool_ocr

EXPOSE 80


