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

# Set up Google Generative AI model with API key
api_key = st.secrets["OPENAI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

recognizer = sr.Recognizer()

# Get response from Gemini model (Generative AI)
def get_gemini_response(prompt, history=[]):
    try:
        full_prompt = "\n".join(history) + "\n" + prompt if history else prompt
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

# Summarize text using Gemini model
def summarize_text(text):
    prompt = f"Summarize the following text:\n\n{text}"
    return get_gemini_response(prompt)

# Analyze image using Gemini model
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

# Read text from uploaded document
def read_document(file):
    if "text/plain" in file.type:
        return file.getvalue().decode("utf-8")
    elif "application/pdf" in file.type:
        return "".join(page.extract_text() for page in PyPDF2.PdfReader(file).pages)
    elif "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in file.type:
        return "\n".join(paragraph.text for paragraph in Document(file).paragraphs)
    return "Unsupported file type."

# Chat interface
def chat_interface():
    st.title("Chatbot Interface")
    history = []
    user_input = st.text_input("You:")
    if st.button("Send"):
        if user_input.lower().startswith("summarize:"):
            response = summarize_text(user_input[len("summarize:"):].strip())
        else:
            response = get_gemini_response(user_input, history)
        st.write("Chatbot:", response)
        history.extend(["You: " + user_input, "Chatbot: " + response])

# Image analysis interface
def image_analysis_interface():
    st.title("Image Analysis Interface")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file and st.button("Analyze Image"):
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        prompt = st.text_input("Enter your prompt:", "Describe this image.")
        analysis_result = analyze_image(uploaded_file, prompt)
        st.write("Analysis Result:", analysis_result)

# Document summarization interface
def document_summarization_interface():
    st.title("Document Summarization")
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
    if uploaded_file and st.button("Summarize"):
        doc_text = read_document(uploaded_file)
        summary = summarize_text(doc_text)
        st.write("Summary:", summary)

# Define a class to process audio with streamlit-webrtc
# class AudioProcessor(AudioProcessorBase):
#     def __init__(self):
#         self.recognizer = sr.Recognizer()
#         self.audio_queue = queue.Queue()

#     def recv(self, frame):
#         # Convert WebRTC audio to NumPy array
#         audio_data = np.frombuffer(frame.to_ndarray(), np.float32)
#         self.audio_queue.put(audio_data)
#         return frame

# Voice assistant interface using real-time audio streaming
# def voice_assistant_interface():
#     st.title("Voice Assistant Chatbot")
    
#     # WebRTC streamer for capturing real-time audio
#     webrtc_ctx = webrtc_streamer(
#         key="voice-assistant",
#         mode=WebRtcMode.SENDRECV,
#         audio_processor_factory=AudioProcessor,
#         media_stream_constraints={"audio": True, "video": False},
#     )

#     # Check if the streamer is playing and process audio
#     if webrtc_ctx.state.playing:
#         st.write("Listening...")  # This should display once the audio stream starts
        
#         # Access audio processor
#         audio_processor = webrtc_ctx.audio_processor
#         if audio_processor:
#             st.write("Audio processor is active")  # To check if audio processor is created

#             while not audio_processor.audio_queue.empty():
#                 try:
#                     st.write("Processing audio...")  # Add a log to check if audio is being processed
#                     # Capture and transcribe audio data
#                     audio_data = audio_processor.audio_queue.get()
#                     audio = sr.AudioData(audio_data.tobytes(), 16000, 2)
#                     transcription = recognizer.recognize_google(audio)

#                     # Display transcription result
#                     st.write(f"You: {transcription}")

#                     # Generate response from Gemini AI
#                     response = get_gemini_response(transcription)
#                     st.write(f"Chatbot: {response}")

#                 except sr.UnknownValueError:
#                     st.write("Sorry, I couldn't understand the audio.")
#                 except sr.RequestError as e:
#                     st.write(f"Error with recognition service: {e}")
#         else:
#             st.write("Audio processor is not active.")
#     else:
#         st.write("WebRTC is not streaming audio.")

# Streamlit sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Chatbot", "Image Analysis", "Document Summarization"])

if page == "Chatbot":
    chat_interface()
elif page == "Image Analysis":
    image_analysis_interface()
elif page == "Document Summarization":
    document_summarization_interface()
# elif page == "Voice Assistant":
#     voice_assistant_interface()
