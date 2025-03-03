import google.generativeai as genai
import base64
import os
import streamlit as st
from PIL import Image
import io
import PyPDF2  # For PDF files
from docx import Document  # For DOCX files
import speech_recognition as sr
#import pyttsx3
import re
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import speech_recognition as sr
import numpy as np
import queue

api_key = st.secrets["OPENAI_API_KEY"]

# Configure Gemini API
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Text-to-speech setup
#engine = pyttsx3.init()

# Speech-to-text setup
recognizer = sr.Recognizer()

text = "some text"
pattern = r"find_me"  # This pattern won't match

match = re.search(pattern, text)

if match:
    result = match.group(0)  # Access the match
    print(result)
else:
    print("No match found")

def get_gemini_response(prompt, history=[]):
    system_prompt = "You are a helpful and informative chatbot. Answer questions clearly and concisely."
    try:
        if history:
            full_prompt = "\n".join(history) + "\n" + prompt
        else:
            full_prompt = prompt
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def summarize_text(text):
    prompt = f"Summarize the following text:\n\n{text}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def analyze_image(image_file, prompt):
    try:
        img = Image.open(image_file)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_data = {"mime_type": "image/jpeg", "data": encoded_image}

        contents = [
            prompt,
            image_data
        ]

        response = model.generate_content(contents)
        return response.text
    except Exception as e:
        return f"An image analysis error occurred: {e}"

def read_document(file):
    file_type = file.type
    if "text/plain" in file_type:
        return file.getvalue().decode("utf-8")
    elif "application/pdf" in file_type:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in file_type:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    else:
        return "Unsupported file type."

def document_summarization_interface():
    st.title("Document Summarization")
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        with st.spinner("Summarizing..."):
            document_text = read_document(uploaded_file)
            if "Unsupported file type" not in document_text:
                summary = summarize_text(document_text)
                st.write("Summary:")
                st.write(summary)
            else:
                st.write(document_text)

def chat_interface():
    st.title("Chatbot Interface")
    st.write("Welcome to the Q&A Chatbot! Type 'summarize:' followed by text to summarize.")
    history = []
    user_input = st.text_input("You:")
    if st.button("Send"):
        if user_input.lower() in ["quit", "exit", "bye"]:
            st.write("Chatbot: Goodbye!")
        elif user_input.lower().startswith("summarize:"):
            text_to_summarize = user_input[len("summarize:"):].strip()
            summary = summarize_text(text_to_summarize)
            st.write("Chatbot Summary:", summary)
            history.append("You: " + user_input)
            history.append("Chatbot: " + summary)
        else:
            response = get_gemini_response(user_input, history)
            st.write("Chatbot:", response)
            history.append("You: " + user_input)
            history.append("Chatbot: " + response)

def image_analysis_interface():
    st.title("Image Analysis Interface")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        prompt = st.text_input("Enter your prompt:", "Describe this image.")
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                result = analyze_image(uploaded_file, prompt)
                st.write("Analysis Result:")
                st.write(result)

# def speak(text):
#     try:
#        # Say the text
#         engine.say(text)

#         # Run the speech event loop, if it's not already running
#         engine.runAndWait()
    
#     except RuntimeError:
#         # Handle the case where the loop is already running
#         print("Warning: The speech engine is already running. Skipping 'runAndWait' this time.")
    
# def listen():
#     with sr.Microphone() as source:
#         st.write("Listening... (you have 5 seconds)")
#         audio = recognizer.listen(source, timeout=5)  # Add timeout to prevent waiting indefinitely
#         try:
#             text = recognizer.recognize_google(audio)
#             st.write(f"You said: {text}")
#             return text
#         except sr.UnknownValueError:
#             st.write("Could not understand audio")
#             return None
#         except sr.RequestError as e:
#             st.write(f"Could not request results; {e}")
#             return None


# def voice_assistant_interface():
#     st.title("Voice Assistant Chatbot")
#     history = []

#     if st.button("Start Listening"):
#         user_input = listen()
#         if user_input:
#             response = get_gemini_response(user_input, history)
#             st.write(f"Chatbot: {response}")
#             speak(response)
#             history.append(f"You: {user_input}")
#             history.append(f"Chatbot: {response}")
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.transcription = None  # Store transcribed text

    def recv(self, frame):
        # Convert audio frame to bytes for recognition
        audio_data = np.frombuffer(frame.to_ndarray().flatten(), dtype=np.int16)
        self.audio_queue.put(audio_data)

        # Process audio for speech recognition
        try:
            audio_bytes = b"".join([np.int16(a).tobytes() for a in list(self.audio_queue.queue)])
            audio_source = sr.AudioData(audio_bytes, sample_rate=16000, sample_width=2)

            self.transcription = self.recognizer.recognize_google(audio_source)
            print("Recognized Speech:", self.transcription)  # Debugging
        except sr.UnknownValueError:
            self.transcription = "Could not understand audio"
        except sr.RequestError as e:
            self.transcription = f"Error with speech recognition: {e}"

        return frame


def voice_assistant_interface():
    st.title("Real-time Voice Assistant Chatbot")
    st.write("Speak into the microphone to interact with the chatbot.")
    
    # Start the WebRTC audio streaming
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True},
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        processor = webrtc_ctx.audio_processor
        
        if processor and processor.transcription:
            st.write(f"You said: {processor.transcription}")
            
            # Get response from Gemini
            response = get_gemini_response(processor.transcription)
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
