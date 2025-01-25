import streamlit as st
import sounddevice as sd
import wave
import numpy as np
import pickle
import os
from scipy.io.wavfile import write
from pydub import AudioSegment
import speech_recognition as sr
import nltk
import pickle
import pandas as pd
import re              # package for importing regular expression
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('stopwords')
# nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
from contractions import fix
import os
folder_path = r"D:\Fraud_Call_Detection"
os.makedirs(folder_path, exist_ok=True)

# Load the trained model (replace 'Fraud_Model.pkl' with your actual model)
with open("Fraud_Model.pkl", "rb") as f:
    tfidfvectorizer, nbclassifier, le = pickle.load(f)

def audio_to_text(file_path):
    conv=[]
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language="en-HK")
        conv.append(text)
    except sr.UnknownValueError:
        print("Sorry, I could not understand.")
    return conv

# Method to preprocessing the data
def preprocessing(dataset, num_of_rows=1):
    stemmer = WordNetLemmatizer()
    corpus = []

    for i in range(0, num_of_rows):
        document = fix(dataset[i])  # Expand contractions
        document = re.sub(r'\W', ' ', document)  # Remove special characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)  # Remove single characters
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)  # Remove single char from start
        document = re.sub(r'\s+', ' ', document, flags=re.I)  # Remove extra spaces
        document = document.lower()
        document = ' '.join([stemmer.lemmatize(w) for w in document.split()])
        corpus.append(document)

    return corpus

def process_audio(file_path):
    # Convert wav to text features (placeholder - replace with actual processing)
    audio = AudioSegment.from_wav(file_path)
    text_data = audio_to_text(file_path)  # Implement speech-to-text here
    text_tfidf = tfidfvectorizer.transform(text_data)
    prediction = nbclassifier.predict(text_tfidf)
    predicted_label = le.inverse_transform(prediction)
    st.success(text_data)
    st.success(f"Detected as: {predicted_label[0]}")

def upload_file():
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file:
        file_path = r"D:\Fraud_Call_Detection\uploaded_audio.wav"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        process_audio(file_path)
        os.remove(file_path)

def record_audio():
    duration = 5  # seconds
    fs = 44100  # Sample rate
    st.info("Recording will start. Speak now...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype=np.int16)
    sd.wait()
    save_path = r"D:\Fraud_Call_Detection\recorded_audio.wav"
    write(save_path, fs, recording)
    process_audio(save_path)
    os.remove(save_path)

# Streamlit UI
st.title("Fraud Call Detection System")
st.sidebar.header("This is an app to detect Fraud and Real Calls.\
 It would be helpful in detecting potential fraud calls which are received and causing into financial losses")

# File uploader
if st.button("Upload WAV File"):
    upload_file()

# To record voice
if st.button("Record Audio"):
    record_audio()
