from flask import Flask, render_template, redirect, url_for, send_from_directory, request
from transformers import AutoTokenizer, AutoModel
import os


# Set cache directory to C: or D:
os.environ["TRANSFORMERS_CACHE"] = "C:\\Users\\sri vijaya lakshmi\\.cache\\huggingface"

# Load the tokenizer model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

app = Flask(__name__, template_folder="templates")


@app.route('/')
def home():
    return render_template('front.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('index'))  
    return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/textgeneration')
def textgeneration():
    return render_template('textgeneration.html')

@app.route('/imagegeneration')
def imagegeneration():
    return render_template('imagegeneration.html')

@app.route('/videogeneration')
def videogeneration():
    return render_template('videogeneration.html')

@app.route('/moretools')
def toolgeneration():
    return render_template('moretools.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == '__main__':
    app.run(debug=True)
