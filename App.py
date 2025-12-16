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
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize Flask app and OpenAI client
app = Flask(__name__)
CORS(app)  # Enable CORS for React Native access
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory stores
documents = []
embeddings = []
index = None

# --- Embedding utilities ---
def embed_texts(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [np.array(e.embedding, dtype="float32") for e in response.data]

def add_to_index(chunks):
    global documents, embeddings, index
    new_embeddings = embed_texts(chunks)
    if index is None:
        dim = len(new_embeddings[0])
        index = faiss.IndexFlatL2(dim)
    index.add(np.array(new_embeddings))
    documents.extend(chunks)
    embeddings.extend(new_embeddings)

# --- Extraction utilities ---
def extract_text_from_image(file_path):
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    return text

def extract_text(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_path.endswith(".xlsx") or file_path.endswith(".csv"):
        df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)
        text = df.to_string()
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        text = extract_text_from_image(file_path)
    return text

# --- Image understanding (GPT-4o multimodal) ---
def ask_about_image(file_path, question):
    with open(file_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }
        ]
    )
    return response.choices[0].message.content

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    file = request.files["file"]
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    os.makedirs('./temp', exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir='./temp') as tmp:
        file_path = tmp.name + "_" + file.filename
        file.save(file_path)

    text = extract_text(file_path)
    if not text.strip():
        return jsonify({"message": "No readable text found"}), 400
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    add_to_index(chunks)
    return jsonify({"message": f"File {file.filename} uploaded and indexed."})

@app.route("/ask", methods=["POST"])
def ask():
    global index, documents
    data = request.json
    query = data.get("query", "")
    if index is None:
        return jsonify({"answer": "No documents uploaded yet."}), 400

    query_vec = embed_texts([query])[0].reshape(1, -1)
    D, I = index.search(query_vec, 3)
    retrieved = [documents[i] for i in I[0]]

    context = "\n".join(retrieved)
    # Updated prompt to generate only questions and options, without answers or explanations
    prompt = f"Based on the following context:\n{context}\n\n{query}\n\nGenerate only the requested questions and options (e.g., for MCQ or True/False). Do not include any answers, explanations, or additional text. For fill in the blanks and short answer questions, generate only the questions. Format each question on its own line, with options listed below if applicable."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return jsonify({"answer": response.choices[0].message.content})

@app.route("/ask_image", methods=["POST"])
def ask_image():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    file = request.files["file"]
    question = request.form.get("question", "Describe this image")
    os.makedirs('./temp', exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir='./temp') as tmp:
        file_path = tmp.name + "_" + file.filename
        file.save(file_path)

    answer = ask_about_image(file_path, question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from local network
