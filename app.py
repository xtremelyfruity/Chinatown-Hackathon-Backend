import os
import io
import cv2
import json
import requests
import numpy as np
from PIL import Image
import pytesseract
import tempfile
import whisper  # pip install git+https://github.com/openai/whisper.git
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests if React is on a different port

# -----------------------------
# 1. Configuration and Settings
# -----------------------------

# Tesseract path (update as needed)
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\19253\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

# API Keys (use environment variables in production)
ANTHROPIC_API_KEY = os.getenv(
    "ANTHROPIC_API_KEY",
    "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"
)
TWELVE_LABS_API_KEY = os.getenv(
    "TWELVE_LABS_API_KEY",
    "tlk_398YQW921F6JBK2ARCPKG34FJF87"
)
TWELVE_LABS_INDEX_ID = os.getenv(
    "TWELVE_LABS_INDEX_ID",
    "67d70b3895d40cc75f6a64db"
)

# Allowed file extensions for OCR (images) and speech (audio)
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "bmp", "gif"}
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "m4a", "ogg"}

# -----------------------------
# 2. Load Models
# -----------------------------
# Load the small Whisper model for speech-to-text (supports English, Chinese, etc.)
whisper_model = whisper.load_model("small")

# -----------------------------
# 3. Helper Functions
# -----------------------------

def allowed_file(filename, allowed_set):
    """Check if the file has an allowed extension from the provided set."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

def preprocess_image(image_np):
    """
    Convert a grayscale image (NumPy array) to a thresholded image
    for better OCR performance.
    """
    blurred = cv2.GaussianBlur(image_np, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )
    return thresh

def build_claude_messages(ocr_text: str, user_input: str = ""):
    """
    Returns a list of message objects suitable for Anthropic's Messages API.
    We'll keep it simple: system instructions plus user content.
    """
    

    if user_input.strip():
        user_content = (
            f"The user has provided the following text from a federal document:\n{ocr_text}\n\n"
            f"They also wrote this additional prompt:\n{user_input}\n"
        )
    else:
        user_content = (
            f"The user has provided the following text from a federal document:\n{ocr_text}\n"
        )

    return [
        {"role": "user", "content": user_content}
    ]

def call_anthropic_claude_messages(messages):
    """
    Sends messages to Anthropic's Claude using the v1/messages endpoint
    (for Claude 3 models and newer).
    """
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": "claude-3-sonnet-20240229",  # Or another Claude 3 model
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.7,
        "system" : "You are a helpful assistant specialized in analyzing federal documents. Explain what the document is about and how to fill it out."
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Anthropic API Error: {response.status_code} {response.text}")

def call_twelve_labs_video_analysis(video_path):
    """
    Calls Twelve Labs API to analyze the provided video file.
    Ensure that your environment has TWELVE_LABS_API_KEY and TWELVE_LABS_INDEX_ID set.
    """
    TASKS_URL = "https://api.twelvelabs.io/v1/tasks"
    headers = {"x-api-key": TWELVE_LABS_API_KEY}
    data = {
        "index_id": TWELVE_LABS_INDEX_ID,
        "language": "en"
    }
    with open(video_path, "rb") as f:
        files = {
            "video_file": (os.path.basename(video_path), f, "application/octet-stream")
        }
        response = requests.post(TASKS_URL, headers=headers, data=data, files=files)
        if response.status_code == 201:
            return response.json()  # Contains details about the created task
        else:
            raise ValueError(f"Twelve Labs API Error: {response.status_code} {response.text}")

# -----------------------------
# 4. Endpoints
# -----------------------------

@app.route('/ocr', methods=['POST'])
def extract_text_and_call_claude():
    """
    Endpoint for processing an uploaded file or text-only:
      - If an image file is uploaded: runs Tesseract OCR, then calls Anthropic Claude-2.
      - If an MP4 video is uploaded: sends the full video file to Twelve Labs for analysis.
      - If no file is provided: sends the user's text alone to Claude.
    """
    print("----- /ocr Request Debug Info -----")
    print("request.files keys:", list(request.files.keys()))
    print("request.form:", request.form)
    print("-------------------------------------")

    user_input = request.form.get('message', '').strip()
    file = request.files.get('file', None)

    # If no file is provided or filename is empty, process text-only.
    if not file or file.filename == '':
        try:
            messages = build_claude_messages("", user_input)
            claude_response = call_anthropic_claude_messages(messages)
            return jsonify({
                "type": "text_only",
                "message": user_input,
                "claude_response": claude_response
            })
        except Exception as e:
            return jsonify({
                "error": "Failed to process text-only request",
                "details": str(e)
            }), 500

    file_extension = file.filename.rsplit('.', 1)[1].lower()

    # Video file scenario: process MP4 using Twelve Labs API
    if file_extension == "mp4":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(file.read())
            video_path = tmp_video.name
        try:
            twelve_labs_response = call_twelve_labs_video_analysis(video_path)
            return jsonify({
                "type": "video_analysis",
                "twelve_labs_response": twelve_labs_response
            })
        except Exception as e:
            return jsonify({
                "error": "Failed to analyze video with Twelve Labs",
                "details": str(e)
            }), 500
        finally:
            try:
                os.remove(video_path)
            except Exception:
                pass

    # Image file scenario: run Tesseract OCR and call Claude
    elif allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        try:
            # Read file contents once
            file_data = file.read()
            image_pil = Image.open(io.BytesIO(file_data)).convert("L")
            image_np = np.array(image_pil)
            processed_image = preprocess_image(image_np)
            ocr_text = pytesseract.image_to_string(processed_image).strip()
            print("OCR Text:", ocr_text)
            print("USER INPUT", user_input)

            # Build messages using the OCR text and any additional user input
            messages = build_claude_messages(ocr_text, user_input)
            print(messages)
            claude_response = call_anthropic_claude_messages(messages)

            print("----- OCR & Claude Response -----")
            print("Messages:\n", json.dumps(messages, indent=2))
            print("Claude Response:\n", json.dumps(claude_response, indent=2))
            print("----------------------------------")

            return jsonify({
                "type": "ocr",
                "ocr_text": ocr_text,
                "claude_response": claude_response
            })
        except Exception as e:
            return jsonify({
                "error": "Failed to process image",
                "details": str(e)
            }), 500
    else:
        return jsonify({"error": "Unsupported file format for OCR"}), 400

@app.route('/speak', methods=['POST'])
def speak_text():
    """
    Endpoint for text-to-speech conversion:
      - This endpoint is disabled for now.
    """
    return jsonify({"error": "Endpoint disabled"}), 404

@app.route('/mic', methods=['POST'])
def transcribe_audio():
    """
    Endpoint for speech-to-text:
      - Accepts an audio file.
      - Uses the Whisper model to transcribe the audio.
      - Returns the transcript and detected language.
    """
    print("----- /mic Request Debug Info -----")
    print("request.files keys:", list(request.files.keys()))
    print("request.form:", request.form)
    print("-------------------------------------")

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        return jsonify({"error": "Unsupported audio file format"}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as temp_audio:
            temp_audio_path = temp_audio.name
            file.save(temp_audio_path)

        result = whisper_model.transcribe(temp_audio_path)
        transcript = result.get("text", "").strip()
        detected_language = result.get("language", "unknown")

        print("----- Audio Transcription -----")
        print("Detected Language:", detected_language)
        print("Transcript:", transcript)
        print("--------------------------------")

        return jsonify({
            "type": "mic",
            "language": detected_language,
            "transcript": transcript
        })
    except Exception as e:
        return jsonify({
            "error": "Failed to transcribe audio",
            "details": str(e)
        }), 500
    finally:
        try:
            os.remove(temp_audio_path)
        except Exception:
            pass

# -----------------------------
# 5. Main
# -----------------------------
if __name__ == '__main__':
    # For local debugging only. In production, use a proper WSGI server.
    app.run(debug=True, host='0.0.0.0', port=5000)
