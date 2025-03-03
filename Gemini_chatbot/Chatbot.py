import google.generativeai as genai
import base64
import os
import streamlit as st
from PIL import Image
import io
import PyPDF2  # For PDF files
from docx import Document  # For DOCX files
import speech_recognition as sr
import re
import numpy as np
import queue
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

api_key = st.secrets["OPENAI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-pro-latest')
recognizer = sr.Recognizer()

def get_gemini_response(prompt, history=[]):
    try:
        full_prompt = "\n".join(history) + "\n" + prompt if history else prompt
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def summarize_text(text):
    prompt = f"Summarize the following text:\n\n{text}"
    return get_gemini_response(prompt)

def analyze_image(image_file, prompt):
    try:
        img = Image.open(image_file).convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": encoded_image}])
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def read_document(file):
    if "text/plain" in file.type:
        return file.getvalue().decode("utf-8")
    elif "application/pdf" in file.type:
        return "".join(page.extract_text() for page in PyPDF2.PdfReader(file).pages)
    elif "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in file.type:
        return "\n".join(paragraph.text for paragraph in Document(file).paragraphs)
    return "Unsupported file type."

def chat_interface():
    st.title("Chatbot Interface")
    history, user_input = [], st.text_input("You:")
    if st.button("Send"):
        response = summarize_text(user_input[len("summarize:"):].strip()) if user_input.lower().startswith("summarize:") else get_gemini_response(user_input, history)
        st.write("Chatbot:", response)
        history.extend(["You: " + user_input, "Chatbot: " + response])

def image_analysis_interface():
    st.title("Image Analysis Interface")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file and st.button("Analyze Image"):
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("Analysis Result:", analyze_image(uploaded_file, st.text_input("Enter your prompt:", "Describe this image.")))

def document_summarization_interface():
    st.title("Document Summarization")
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
    if uploaded_file and st.button("Summarize"):
        st.write("Summary:", summarize_text(read_document(uploaded_file)))

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()  # Queue to store audio frames
        self.transcription = None  # To store the transcription

    def recv(self, frame):
        # Collect audio frames and convert them into recognizable format
        audio_data = np.frombuffer(frame.to_ndarray(), dtype=np.int16).tobytes()
        self.audio_queue.put(audio_data)

        # Perform speech recognition on the collected frames
        if self.audio_queue.qsize() > 20:  # Start recognizing after receiving enough frames
            try:
                audio_bytes = b"".join(list(self.audio_queue.queue))
                audio_source = sr.AudioData(audio_bytes, sample_rate=16000, sample_width=2)
                self.transcription = self.recognizer.recognize_google(audio_source)
            except sr.UnknownValueError:
                self.transcription = "Could not understand audio"
            except sr.RequestError as e:
                self.transcription = f"Error: {e}"

        return frame

def voice_assistant_interface():
    st.title("Voice Assistant Chatbot")
    
    # Set up WebRTC streamer for capturing audio
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,  # Use custom audio processor
        media_stream_constraints={"audio": True},  # Enable audio stream
        async_processing=True
    )
    
    # If audio streaming is active and transcription is available
    if webrtc_ctx.state.playing and webrtc_ctx.audio_processor.transcription:
        st.write(f"You: {webrtc_ctx.audio_processor.transcription}")
        
        # Get the chatbot response based on the recognized transcription
        response = get_gemini_response(webrtc_ctx.audio_processor.transcription)
        st.write(f"Chatbot: {response}")


st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Chatbot", "Image Analysis", "Document Summarization", "Voice Assistant"])

if page == "Chatbot":
    chat_interface()
elif page == "Image Analysis":
    image_analysis_interface()
elif page == "Document Summarization":
    document_summarization_interface()
elif page == "Voice Assistant":
    voice_assistant_interface()
