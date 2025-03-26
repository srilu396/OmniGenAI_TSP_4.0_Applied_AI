from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file, send_from_directory, render_template
from gtts import gTTS
import time
import os
import re
import pdfplumber
import docx
import io
from rembg import remove
from deep_translator import GoogleTranslator
from flask_cors import CORS
import shutil
from werkzeug.utils import secure_filename
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
import google.generativeai as genai
from PIL import Image, ImageEnhance
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
import torch
import requests
import cv2
import numpy as np

# **Fix: Initialize Flask App**

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable Cross-Origin Resource Sharing

# text_to_image.py
API_KEY = "api-key"
API_URL = "api-url"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

def generate_image(prompt):
    """
    Sends a request to Hugging Face API to generate an image.
    Optimized for faster response with minimal retries.
    """
    payload = {"inputs": prompt}
    retries = 3  # Reduced retries for faster failover
    wait_time = 5  # Initial wait time reduced for faster attempts
    timeout_duration = 50  # Set a max timeout of 50 seconds for response

    for attempt in range(retries):
        try:
            print(f"🚀 Attempt {attempt + 1}/{retries}: Generating image for prompt: '{prompt}'")
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=timeout_duration)

            if response.status_code == 200:
                image_path = "static/generated_image.png"
                with open(image_path, "wb") as file:
                    file.write(response.content)
                print(f"✅ Image generated successfully: {image_path}")
                return image_path

            elif response.status_code == 503:
                print(f"⏳ Model is loading... Retrying in {wait_time} seconds (Attempt {attempt + 1}/{retries})")
                time.sleep(wait_time)
                wait_time += 2  # Slightly increase time for next retry

            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                return None  # Fail fast on other errors

        except requests.Timeout:
            print(f"⚠️ Timeout Error: API did not respond within {timeout_duration} seconds")
            return None  # Fail if API takes too long

        except requests.RequestException as e:
            print(f"❌ Network Error: {e}")
            time.sleep(wait_time)

    print("❌ Failed to generate image after retries")
    return None

# image_to_text.py
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_caption(img_path):
    """Generates a description of an image."""
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    output = model.generate(**inputs)
    caption = processor.batch_decode(output, skip_special_tokens=True)[0]

    return caption
    
#image_to_image.py
A = "uploads"
B = "output"
os.makedirs(A, exist_ok=True)
os.makedirs(B, exist_ok=True)

# Dictionary of available effects
EFFECTS = {
    "sketch": lambda img: cv2.cvtColor(cv2.divide(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                                  255 - cv2.GaussianBlur(255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                                                         (21, 21), 0), scale=256),
                                       cv2.COLOR_GRAY2BGR),
    "watercolor": lambda img: cv2.stylization(cv2.bilateralFilter(img, 9, 75, 75), sigma_s=150, sigma_r=0.25),
    "hd": lambda img: cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])),
    "oil_painting": lambda img: cv2.xphoto.oilPainting(img, 7, 1),
    "neon_glow": lambda img: cv2.addWeighted(img, 0.7, cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR), 0.3, 0),
    "3d_effect": lambda img: cv2.merge((img[:, :, 0], img[:, :, 1], np.roll(img[:, :, 2], 10, axis=1))),
    "cartoon": lambda img: cv2.bitwise_and(cv2.bilateralFilter(img, 9, 250, 250),
                                           cv2.bilateralFilter(img, 9, 250, 250),
                                           mask=cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255,
                                                                      cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9))
}

# hand.py
GEMINI_API_KEY = "AIzaSyB1utc2X3afAOTFY_XsHHO7WS8ESZDnVAo"
genai.configure(api_key=GEMINI_API_KEY)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route('/')
@app.route('/imagegeneration')
def imagegeneration():
    return render_template('imagegeneration.html')

@app.route('/text_to_image')
def text_to_image():
    return render_template('text_to_image.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    image_path = generate_image(prompt)

    if image_path:
        return jsonify({"image_url": "/" + image_path})  # Correct relative URL
    else:
        return jsonify({"error": "Image generation failed"}), 500
    
@app.route('/image_to_text', methods=['GET', 'POST'])
def image_to_text():
    return render_template('image_to_text.html')

@app.route('/upload_images', methods=['POST'])
def upload_images():
    """Handle image upload and generate caption."""
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    # Generate caption
    try:
        caption = generate_caption(file_path)
        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the saved file
        os.remove(file_path)

@app.route('/image_to_image')
def image_to_image():
    return render_template('image_to_image.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handles image upload and effect application."""
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded!"}), 400

    file = request.files['image']
    effect = request.form.get("effect")

    if file.filename == '' or effect not in EFFECTS:
        return jsonify({"error": "Invalid input!"}), 400

    img_path = os.path.join(A, file.filename)
    file.save(img_path)

    img = cv2.imread(img_path)
    if img is None:
        return jsonify({"error": "Invalid image file!"}), 400

    processed_img = EFFECTS[effect](img)
    output_path = os.path.join(B, "processed_" + file.filename)
    cv2.imwrite(output_path, processed_img)

    return send_file(output_path, mimetype='image/jpeg')

@app.route('/background')
def background():
    return render_template('background.html')

@app.route('/remove_background', methods=['GET', 'POST'])
def remove_bg():
    image_file = request.files['image']
    color = request.form.get('color', 'Transparent')
    gradient = request.form.get('gradient', 'None')

    image = Image.open(image_file).convert('RGBA')
    output = remove(image)

    if gradient != 'None':
        color1, color2 = gradient.split('-')
        gradient_bg = create_linear_gradient(output.size, color1, color2)
        output = Image.alpha_composite(gradient_bg, output)
    elif color != 'Transparent':
        bg = Image.new('RGBA', output.size, color)
        output = Image.alpha_composite(bg, output)

    byte_io = io.BytesIO()
    output.save(byte_io, 'PNG')
    byte_io.seek(0)
    return send_file(byte_io, mimetype='image/png')


# Function to create gradient
def create_linear_gradient(size, color1, color2):
    width, height = size
    base = Image.new('RGBA', size, color1)
    top = Image.new('RGBA', size, color2)
    mask = Image.new('L', size)
    mask_data = []

    for y in range(height):
        mask_data.extend([int(255 * (y / height))] * width)

    mask.putdata(mask_data)
    gradient = Image.composite(top, base, mask)
    return gradient

@app.route('/hand')
def hand():
    return render_template('hand.html')

def allowed_file(filename):
    """Check if the uploaded file is an allowed image format"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def process_handwriting_with_gemini(image_path):
    """Send image to Gemini 1.5 API for handwriting recognition"""
    try:
        # Open image using PIL
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()

        image = Image.open(io.BytesIO(image_data))

        # Use Gemini 1.5 with a text prompt
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(["Extract the text from this handwritten note.", image])

        # Extract and return the text response
        extracted_text = response.text.strip()
        return extracted_text
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def hand_index():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded!"})

        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format! Use PNG, JPG, or JPEG."})

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        extracted_text = process_handwriting_with_gemini(file_path)

        return jsonify({"text": extracted_text, "image_url": file_path})


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == '__main__':
    app.run(debug=True)