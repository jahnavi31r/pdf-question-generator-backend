import os
import tempfile
import base64
import pdfplumber
import docx
import pandas as pd
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app and OpenAI client
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Get API key safely
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    app.logger.error("OPENAI_API_KEY not set in environment variables!")
    raise ValueError("OPENAI_API_KEY is required")

client = OpenAI(api_key=api_key)

# In-memory stores (resets on restart — expected for simple app)
documents = []
index = None

# --- Embedding utilities ---
def embed_texts(texts):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [np.array(e.embedding, dtype="float32") for e in response.data]
    except Exception as e:
        app.logger.error(f"Embedding error: {str(e)}")
        raise

def add_to_index(chunks):
    global documents, index
    new_embeddings = embed_texts(chunks)
    if index is None:
        dim = len(new_embeddings[0])
        index = faiss.IndexFlatL2(dim)
    index.add(np.array(new_embeddings))
    documents.extend(chunks)
    app.logger.info(f"Added {len(chunks)} chunks to index. Total documents: {len(documents)}")

# --- Extraction utilities ---
def extract_text_from_image(file_path):
    try:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        app.logger.error(f"OCR error: {str(e)}")
        return ""

def extract_text(file_path):
    text = ""
    try:
        if file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
            text = df.to_string()
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            text = df.to_string()
        elif file_path.endswith((".png", ".jpg", ".jpeg")):
            text = extract_text_from_image(file_path)
    except Exception as e:
        app.logger.error(f"Text extraction error: {str(e)}")
    return text

# --- Image understanding ---
def ask_about_image(file_path, question):
    try:
        with open(file_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }]
        )
        return response.choices[0].message.content
    except Exception as e:
        app.logger.error(f"Image query error: {str(e)}")
        raise

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"message": "No file part"}), 400
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400

        # Save temporarily
        os.makedirs('./temp', exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir='./temp', suffix=os.path.splitext(file.filename)[1]) as tmp:
            file_path = tmp.name
            file.save(file_path)

        text = extract_text(file_path)
        if not text.strip():
            os.unlink(file_path)  # Clean up
            return jsonify({"message": "No readable text found in the file"}), 400

        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        add_to_index(chunks)

        # Clean up file after processing
        try:
            os.unlink(file_path)
        except:
            pass

        return jsonify({"message": f"File '{file.filename}' uploaded and indexed successfully."})

    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        return jsonify({"message": f"Upload failed: {str(e)}"}), 500

@app.route("/ask", methods=["POST"])
def ask():
    try:
        global index, documents
        data = request.json
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"answer": "Please provide a query."}), 400
        if index is None or len(documents) == 0:
            return jsonify({"answer": "No documents uploaded yet. Please upload a file first."}), 400

        query_vec = embed_texts([query])[0].reshape(1, -1)
        D, I = index.search(query_vec, 3)
        retrieved = [documents[i] for i in I[0] if i != -1]

        context = "\n".join(retrieved)
        prompt = f"""Based on the following context:\n{context}\n\nUser query: {query}\n\nGenerate only the requested questions with options (for MCQs, True/False, etc.). Do not include answers, explanations, or extra text. Format clearly with one question per section and options below if needed."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({"answer": answer})

    except Exception as e:
        app.logger.error(f"Ask error: {str(e)}")
        return jsonify({"answer": f"Error generating questions: {str(e)}"}), 500

@app.route("/ask_image", methods=["POST"])
def ask_image():
    try:
        if 'file' not in request.files:
            return jsonify({"message": "No file part"}), 400
        file = request.files["file"]
        question = request.form.get("question", "Describe this image in detail")

        os.makedirs('./temp', exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir='./temp', suffix=os.path.splitext(file.filename)[1]) as tmp:
            file_path = tmp.name
            file.save(file_path)

        answer = ask_about_image(file_path, question)

        # Clean up
        try:
            os.unlink(file_path)
        except:
            pass

        return jsonify({"answer": answer})

    except Exception as e:
        app.logger.error(f"Image ask error: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
