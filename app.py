import os
import io
import cv2
import json
import requests
import numpy as np
from PIL import Image
import pytesseract
import tempfile
import whisper  # Make sure to install OpenAI's whisper: pip install git+https://github.com/openai/whisper.git
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
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\19253\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# API Keys (hardcoded here for demonstration; better to use environment variables)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "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")
TWELVE_LABS_API_KEY = os.getenv("TWELVE_LABS_API_KEY", "tlk_398YQW921F6JBK2ARCPKG34FJF87")
TWELVE_LABS_INDEX_ID = os.getenv("TWELVE_LABS_INDEX_ID", "67d70b3895d40cc75f6a64db")

# Allowed file extensions for OCR (images) and speech (audio)
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "bmp", "gif"}
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "m4a", "ogg"}

# -----------------------------
# 2. Load Models
# -----------------------------
# Load the small Whisper model for speech-to-text (supports English, Chinese, and more)
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

def call_anthropic_claude(prompt):
    """
    Sends the prompt to Anthropic Claude’s API and returns the JSON response.
    Includes the required `anthropic-version` header.
    """
    url = "https://api.anthropic.com/v1/complete"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"  # Required by Anthropic
    }
    payload = {
        "prompt": prompt,
        "model": "claude-3-7-sonnet-20250219",  # Use a valid model name
        "max_tokens_to_sample": 300,
        "temperature": 0.7,
        "stop_sequences": ["\n\nHuman:"],
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Anthropic API Error: {response.status_code} {response.text}")

def build_claude_prompt(ocr_text, user_input=""):
    """
    Build a conversation-format prompt for Anthropic, with a system message,
    a human message (containing the OCR text), and an optional user input.
    Ends with 'Assistant:' so Anthropic knows where to continue.
    """
    system_instructions = (
        "System: You are a helpful assistant specialized in analyzing federal documents. "
        "Explain what the document is about and how to fill it out.\n"
    )
    if user_input.strip():
        human_part = (
            f"Human: The user has provided the following text from a federal document:\n"
            f"{ocr_text}\n\n"
            f"They also wrote this additional prompt:\n"
            f"{user_input}\n\n"
        )
    else:
        human_part = (
            f"Human: The user has provided the following text from a federal document:\n"
            f"{ocr_text}\n\n"
        )
    assistant_part = "Assistant:"
    prompt = f"{system_instructions}\n{human_part}\n{assistant_part}"
    return prompt

"""
def call_elevenlabs_tts(text_content):
    # This function has been removed as per requirements.
"""

def call_twelve_labs_video_analysis(video_path):
    """
    Calls Twelve Labs API to analyze the provided video file.
    This function creates a video indexing task on Twelve Labs.
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
             return response.json()  # Contains details about the created task.
         else:
             raise ValueError(f"Twelve Labs API Error: {response.status_code} {response.text}")

# -----------------------------
# 4. Endpoints
# -----------------------------

@app.route('/ocr', methods=['POST'])
def extract_text_and_call_claude():
    """
    Endpoint for processing an uploaded file or text-only:
      - If an image file is uploaded: runs OCR, builds a prompt with optional user text,
        calls Anthropic Claude's API, and returns the combined response.
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
            # Build prompt with no OCR text
            claude_prompt = build_claude_prompt("", user_input)
            claude_response = call_anthropic_claude(claude_prompt)
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

    # Video file scenario: process MP4 using Twelve Labs API.
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

    # Image file scenario: run OCR and call Claude.
    elif allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        try:
            image_pil = Image.open(io.BytesIO(file.read())).convert("L")
            image_np = np.array(image_pil)
            processed_image = preprocess_image(image_np)
            ocr_text = pytesseract.image_to_string(processed_image).strip()
            claude_prompt = build_claude_prompt(ocr_text, user_input)
            claude_response = call_anthropic_claude(claude_prompt)

            print("----- OCR & Claude Response -----")
            print("Prompt:\n", claude_prompt)
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
      - This endpoint has been disabled/removed as per your requirements.
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
    app.run(debug=True, host='0.0.0.0', port=5000)
