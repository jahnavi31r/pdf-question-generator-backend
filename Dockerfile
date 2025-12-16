FROM python:3.11-slim

# Install system dependencies (Tesseract OCR + utilities for pdfplumber)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app
COPY . .

# Expose port
EXPOSE 5000

# Run with gunicorn (better for production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "App:app"]
