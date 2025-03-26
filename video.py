import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, send_file, send_from_directory, jsonify
from flask import render_template
import os
import subprocess
from tkinter import Tk, filedialog
from werkzeug.utils import secure_filename

# Hide Tkinter main window
Tk().withdraw()

app = Flask(__name__)

UPLOAD__FOLDER = "uploads"
PROCESSED__FOLDER = "processed"
os.makedirs(UPLOAD__FOLDER, exist_ok=True)
os.makedirs(PROCESSED__FOLDER, exist_ok=True)

# Initialize Mediapipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

X = "uploads"
Y = "static"
app.config["X"] = X
app.config["Y"] = Y

os.makedirs(X, exist_ok=True)
os.makedirs(Y, exist_ok=True)

uploaded_images = []

@app.route('/')
@app.route('/videogeneration')
def videogeneration():
    return render_template('videogeneration.html')

@app.route('/video_to_text')
def video_to_text():
    return render_template('video_to_text.html')

@app.route('/image_to_video')
def image_to_video():
    return render_template('image_to_video.html')

def animate_frame(img, animation_type, num_frames):
    frames = []
    h, w = img.shape[:2]

    for j in range(num_frames):
        alpha = j / num_frames  # Animation progress (0 to 1)

        if animation_type == "fade":
            frames.append(cv2.addWeighted(img, 1 - alpha, img, alpha, 0))

        elif animation_type == "zoom":
            scale = 1 + alpha * 0.1
            zoomed = cv2.resize(img, (int(w * scale), int(h * scale)))
            zoomed = zoomed[int((zoomed.shape[0] - h) / 2): int((zoomed.shape[0] + h) / 2),
                            int((zoomed.shape[1] - w) / 2): int((zoomed.shape[1] + w) / 2)]
            frames.append(zoomed)

        elif animation_type == "left_right":
            shift = int(alpha * 50)
            M = np.float32([[1, 0, shift], [0, 1, 0]])
            frames.append(cv2.warpAffine(img, M, (w, h)))

        elif animation_type == "up_down":
            shift = int(alpha * 50)
            M = np.float32([[1, 0, 0], [0, 1, shift]])
            frames.append(cv2.warpAffine(img, M, (w, h)))

        elif animation_type == "rotate":
            angle = alpha * 10
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            frames.append(cv2.warpAffine(img, M, (w, h)))

        elif animation_type == "blur":
            blur_value = int(1 + alpha * 10)
            blurred = cv2.GaussianBlur(img, (blur_value | 1, blur_value | 1), 0)
            frames.append(blurred)

        elif animation_type == "color_shift":
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + int(alpha * 50)) % 180
            color_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            frames.append(color_shifted)

        elif animation_type == "shake":
            dx = np.random.randint(-5, 5)
            dy = np.random.randint(-5, 5)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frames.append(cv2.warpAffine(img, M, (w, h)))

        elif animation_type == "wave":
            wave = img.copy()
            for y in range(h):
                shift_x = int(10 * np.sin(2 * np.pi * y / 30 + alpha * 10))
                wave[y] = np.roll(img[y], shift_x, axis=0)
            frames.append(wave)

        elif animation_type == "brightness":
            brightness = alpha * 50
            bright = cv2.convertScaleAbs(img, alpha=1, beta=brightness)
            frames.append(bright)

    return frames

# Function to generate a video from images with the selected animation type
def generate_video(image_paths, output_path="static/output.mp4", animation_type="fade", duration_per_image=3, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))

    valid_images = 0

    for img_path in image_paths:
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ ERROR: Could not read image {img_path}. Skipping...")
            continue

        img = cv2.resize(img, (1280, 720))
        num_frames = fps * duration_per_image
        frames = animate_frame(img, animation_type, num_frames)

        for frame in frames:
            video.write(frame)

        valid_images += 1

    video.release()

    if valid_images == 0:
        print("⚠ No valid images processed. Video not created.")
        return None

    print(f"🎬 Video saved as {output_path}")
    return output_path

# Flask API Routes

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["X"], filename)
    file.save(filepath)

    uploaded_images.append(filepath)
    return jsonify({"message": "Image uploaded successfully", "count": len(uploaded_images)})

@app.route("/process_video", methods=["POST"])
def process_video():
    try:
        # Check if request is JSON
        if not request.is_json:
            return jsonify({"error": "Invalid request format. Expected JSON."}), 415

        data = request.get_json()
        if not data or "animationType" not in data:
            return jsonify({"error": "Missing 'animationType' in request body."}), 400

        animation_type = data["animationType"]

        if len(uploaded_images) < 2:
            return jsonify({"error": "Upload at least 2 images before processing."}), 400

        output_video = os.path.join(app.config["Y"], "output.mp4")
        generate_video(uploaded_images, output_video, animation_type)

        if not os.path.exists(output_video):
            return jsonify({"error": "Video generation failed."}), 500

        return jsonify({"message": "Video created successfully", "video_url": "/static/output.mp4"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/video_background')
def video_background():
    return render_template('video_background.html')

def remove_video_background(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmentation.process(frame_rgb)

        # Create a mask
        mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255  # Binary mask

        # Smooth the mask using Gaussian blur
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Apply mask to the frame
        fg = cv2.bitwise_and(frame, frame, mask=mask)

        # Set background color (black)
        bg = np.zeros_like(frame)

        # Combine foreground and background
        final_frame = np.where(mask[..., None] == 255, fg, bg)

        out.write(final_frame.astype(np.uint8))  # Ensure correct data type

    cap.release()
    out.release()

@app.route("/remove_video_background", methods=["POST"])  # ✅ Renamed Route
def remove_video_background_endpoint():
    if "video" not in request.files:
        return "No file uploaded", 400

    video_file = request.files["video"]
    input_path = os.path.join(UPLOAD__FOLDER, video_file.filename)
    output_path = os.path.join(PROCESSED__FOLDER, "output.mp4")

    video_file.save(input_path)
    remove_video_background(input_path, output_path)

    return send_file(output_path, mimetype="video/mp4")

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory("static", filename)

if __name__ == '__main__':
    app.run(debug=True)
