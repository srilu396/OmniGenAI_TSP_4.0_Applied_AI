from flask import Flask, render_template, request, jsonify, send_from_directory
from gtts import gTTS
import time
import os
import re
import pdfplumber
import docx
from deep_translator import GoogleTranslator
from flask_cors import CORS
from werkzeug.utils import secure_filename
from transformers import pipeline
import google.generativeai as genai
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


# ✅ Load Flan-T5 model as fallback for grammar checking
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# Supported programming languages for code generation
CODE_LANGUAGES = ["Python", "JavaScript", "Java", "C++", "C#", "Swift", "Go", "Rust"]
app = Flask(__name__)
# Story categories
STORY_TYPES = ["Story", "Script", "Dialogue"]
GENRES = ["Fantasy", "Sci-Fi", "Mystery", "Thriller", "Horror", "Romance", "Comedy", "Drama"]
CORS(app)  # Enable CORS for cross-origin requests
# ✅ Configure Google Gemini AI API
GENAI_API_KEY = "AIzaSyB1utc2X3afAOTFY_XsHHO7WS8ESZDnVAo"  # Replace with your actual API key

if not GENAI_API_KEY:
    print("❌ ERROR: Gemini API key is missing! Provide your API key in the script.")
    exit()
genai.configure(api_key=GENAI_API_KEY)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Ensure 'static' folder exists for storing audio files
if not os.path.exists("static"):
    os.makedirs("static")

DOCUMENT_UPLOAD_FOLDER = "uploads"
os.makedirs(DOCUMENT_UPLOAD_FOLDER, exist_ok=True)
app.config["DOCUMENT_UPLOAD_FOLDER"] = DOCUMENT_UPLOAD_FOLDER

# Dictionary mapping language names to codes
LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Chinese": "zh-cn",
    "Japanese": "ja"
}

qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
document_text = ""

# ✅ Function to get a grammar answer from Gemini API
def get_gemini_response(question):
    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")  # Using Gemini Pro
        response = model_gemini.generate_content(question)
        return response.text.strip().split("\n")[0]  # ✅ Return only the first line
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None

# ✅ Function to get a grammar response from Flan-T5 (fallback)
def get_t5_response(question):
    input_text = f"Explain in simple words: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():  # No need for gradients in inference
        outputs = model.generate(**inputs, max_length=100)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ Final function to process the grammar question
def grammar_expert(question):
    gemini_answer = get_gemini_response(question)
    if gemini_answer:
        return gemini_answer  # ✅ Use Gemini answer if available
    return get_t5_response(question)  # ✅ Use Flan-T5 as backup
def ask_gemini(prompt, max_retries=2, wait_time=5):
    """Sends a prompt to Gemini API and returns the generated response with retries."""
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")

            # Request a longer, detailed story using Indian English
            detailed_prompt = (
                prompt +
                " Write in simple Indian English, using clear descriptions, natural dialogues, and engaging storytelling."
                " Ensure it is at least 500 words long with a proper beginning, middle, and end."
            )

            response = model.generate_content(detailed_prompt)  
            return response.text.strip()

        except Exception as e:
            print(f"❌ API Error (Attempt {attempt + 1}):", str(e))
            if attempt < max_retries - 1:
                time.sleep(wait_time)  # Wait before retrying
            else:
                return f"Error: {e}"
def extract_code(response):
    """Extracts code from API response."""
    code_lines = []
    recording = False
    for line in response.split("\n"):
        if "```" in line:
            recording = not recording
            continue
        if recording:
            code_lines.append(line)

    return "\n".join(code_lines).strip() if code_lines else response.strip()
@app.route('/')

@app.route('/textgeneration')
def textgeneration():
    return render_template('textgeneration.html')
# Text Processing Routes
@app.route('/translator')
def translator():
    return render_template('translator.html')
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get("text", "").strip()
    target_lang = data.get("language", "")

    if not text:
        return jsonify({"error": "Please enter text to translate!"}), 400

    if target_lang not in LANGUAGES:
        return jsonify({"error": "Invalid target language selected!"}), 400

    target_lang_code = LANGUAGES[target_lang]

    try:
        translated_text = GoogleTranslator(target=target_lang_code).translate(text)
        return jsonify({"original": text, "translated": translated_text})
    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500


@app.route('/text_to_audio')
def text_to_audio():
    return render_template('text_to_audio.html')
@app.route('/generate', methods=['POST'])
def generate_audio():
    text = request.form.get('text')  # Get input text from the form
    if not text.strip():
        return jsonify({"error": "Text cannot be empty"}), 400

    # Generate unique filename using timestamp
    filename = f"output_{int(time.time())}.mp3"
    filepath = os.path.join("static", filename)

    # Generate and save the audio
    tts = gTTS(text=text, lang='en', tld='com', slow=False)
    tts.save(filepath)

    # Return full static path for the frontend
    return jsonify({"audio_url": f"/static/{filename}"})

@app.route('/document')
def document():
    return render_template('document.html')

# Route to upload a PDF or Word document
@app.route("/upload", methods=["POST"])
def upload_file():
    global document_text
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["DOCUMENT_UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Extract text based on file type
    if filename.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file_path)
    elif filename.endswith(".docx"):
        extracted_text = extract_text_from_docx(file_path)
    else:
        return jsonify({"error": "Invalid file type. Please upload a PDF or DOCX."}), 400

    if extracted_text.strip():
        return jsonify({"message": "File uploaded successfully!"})
    else:
        return jsonify({"error": "Could not extract text from the document."}), 400

# Route to answer questions
@app.route("/ask", methods=["POST"])
def ask():
    global document_text
    if not document_text:
        return jsonify({"answer": "No document is uploaded yet. Please upload a PDF or DOCX."})

    question = request.form.get("question")
    if not question.strip():
        return jsonify({"answer": "Please enter a valid question."})
    
    answer = qa_model(question=question, context=document_text)
    return jsonify({"answer": answer["answer"]})

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    global document_text
    document_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                document_text += extracted_text + "\n"
    return document_text

# Function to extract text from a Word document
def extract_text_from_docx(docx_path):
    global document_text
    document_text = ""
    doc = docx.Document(docx_path)
    for para in doc.paragraphs:
        document_text += para.text + "\n"
    return document_text

@app.route('/colour')
def colour():
    return render_template('colour.html')
# ✅ Function to generate color palette
def generate_color_palette(answers):
    prompt = f"""
    Based on these details:
    1. Website type: {answers.get('websiteType', 'Generic')}
    2. Mood: {answers.get('mood', 'Neutral')}
    3. Preferred main color: {answers.get('mainColor', 'None')}
    4. Brightness preference: {answers.get('brightness', 'Bright')}

    Generate a color palette with 5 colors.
    Provide the color names and HEX codes in this format:

    Example:
    - Red Sunset: #FF5733
    - Ocean Blue: #3498DB
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        if not response.text.strip():
            return []

        # ✅ Extract colors using regex
        color_data = re.findall(r'(.+?):\s*(#[0-9a-fA-F]{6})', response.text)
        return [{"name": name.strip(), "hex": hex_code} for name, hex_code in color_data]

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return []

# ✅ Route to generate color palette
@app.route("/generate_palette", methods=["POST"])
def generate_palette():
    data = request.json
    colors = generate_color_palette(data)
    return jsonify(colors)

@app.route('/grammar')
def grammar():
    return render_template('grammar.html')
@app.route('/ask_grammar', methods=['POST'])
def ask_grammar():
    question = request.form.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please enter a valid grammar question."})
    
    answer = grammar_expert(question)  # ✅ Process the question
    return jsonify({"answer": answer})  # ✅ Send back JSON response
@app.route('/code_generator')
def code_generator_page():
    return render_template('code_generator.html')
@app.route("/generate_code", methods=["POST"])
def generate_code():
    data = request.json
    query = data.get("query", "").strip()
    language = data.get("language", "Python").strip()

    if language not in CODE_LANGUAGES:
        return jsonify({"error": "Unsupported language!"})

    code_prompt = f"Generate a {language} script for: {query}. Provide full working code."
    explanation_prompt = f"Explain this {language} code step by step."

    code_response = ask_gemini(code_prompt)
    extracted_code = extract_code(code_response)

    explanation_response = ask_gemini(explanation_prompt + "\n\n" + extracted_code)

    return jsonify({
        "code": extracted_code,
        "explanation": explanation_response
    })
@app.route('/story')
def story():
    return render_template("story.html", story_types=STORY_TYPES, genres=GENRES)

@app.route("/generate_story", methods=["POST"])
def generate_story():
    # Get the data from the request
    data = request.json
    # Get the story type from the data, defaulting to "Story"
    story_type = data.get("story_type", "Story")
    # Get the genre from the data, defaulting to "Fantasy"
    genre = data.get("genre", "Fantasy")
    # Get the prompt from the data, defaulting to an empty string
    prompt = data.get("prompt", "").strip()

    # Check if the prompt is empty
    if not prompt:
        # Return an error message if the prompt is empty
        return jsonify({"error": "Please enter a story idea!"})

    # Create the AI prompt
    ai_prompt = f"Write a detailed {story_type} in the {genre} genre based on this idea: {prompt}."

    # Request AI-generated content with retries
    story_content = ask_gemini(ai_prompt)

    # Check if there is an error in the story content
    if "Error" in story_content:
        # Print the error message
        print("❌ API Error:", story_content)
        # Return an error message if there is an error
        return jsonify({"error": "Failed to generate story. Please try again later."})

    # Return the story content and story type
    return jsonify({"story": story_content, "story_type": story_type})

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == '__main__':
    app.run(debug=True)