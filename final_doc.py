import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import os
import cv2
import coll
import mediapipe as mp
import numpy as np
from playsound import playsound
import smtplib
import time
import speech_recognition as sr
from PIL import Image
import pytesseract
import pyttsx3
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from googletrans import Translator
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from pdfminer.high_level import extract_text
import docx


# Email setup
def setup_email():
    try:
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login("techprosolutioncse@gmail.com", "vsjnmppsyarjuwkv")
        return s
    except smtplib.SMTPAuthenticationError as e:
        st.error(f"SMTP Authentication Error: {e}")
        return None

# Load data and train RandomForest model
def load_and_train_model():
    local_path = os.path.dirname(os.path.realpath('__file__'))
    file_name = 'data1.csv'  # File of total data
    data_path = os.path.join(local_path, file_name)
    df = pd.read_csv(data_path)

    units_in_data = 28  # Number of units in data
    titles = [f"unit-{i}" for i in range(units_in_data)]
    X = df[titles]
    y = df['letter']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)
    clf = RandomForestClassifier(n_estimators=30)  # Random forest classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
    plt.title('Confusion Matrix of RF', size=15)
    plt.show()

    return clf


# Database interaction functions
def create_user(username, password):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        st.success("User created successfully.")
    except sqlite3.OperationalError as e:
        st.error(f"Database error: {e}")
    finally:
        conn.close()

def authenticate_user(username, password):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = c.fetchone()
    except sqlite3.OperationalError as e:
        st.error(f"Database error: {e}")
        user = None
    finally:
        conn.close()
    return user is not None

# Hand sign prediction setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def get_prediction(image, clf):
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        ImageData = coll.ImageToDistanceData(image, hands)
        DistanceData = ImageData['Distance-Data']
        if len(DistanceData) == 0:
            st.write("Error: DistanceData is empty.")
            return "UNKNOWN"
        prediction = clf.predict([DistanceData])
        return prediction[0]

# Function to translate text
def translate_text(text, target_language):
    translator = Translator()
    try:
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        st.write(f"Error in translation: {e}")
        return text

# Function to handle hand signs with translation option
def handle_hand_signs(SpelledWord, target_language, email_sender):
    if SpelledWord == "HAVE A GREAT DAY!":
        detected_text = "HAVE A GREAT DAY!"
    elif SpelledWord == "EMERGENCY":
        detected_text = "EMERGENCY"
        if email_sender:
            message = MIMEMultipart()
            message['From'] = "techprosolutioncse@gmail.com"
            message['To'] = "abninfotechprojects@gmail.com"
            message['Subject'] = "Emergency Alert"
            body = "EMERGENCY"
            message.attach(MIMEText(body, 'plain'))
            email_sender.sendmail(message['From'], message['To'], message.as_string())
            playsound('3.wav')
    elif SpelledWord == "A CUP OF COFFEE?":
        detected_text = "A CUP OF COFFEE?"
    elif SpelledWord == "HAVE A BREAKFAST!":
        detected_text = "HAVE A BREAKFAST!"
    elif SpelledWord == "GOOD NIGHT! SLEEP WELL.":
        detected_text = "GOOD NIGHT! SLEEP WELL."
    elif SpelledWord == "HOW ARE YOU?":
        detected_text = "HOW ARE YOU?"
    else:
        detected_text = "UNKNOWN"
    
    st.write(f"Detected Hand Sign: {detected_text}")
    translated_text = translate_text(detected_text, target_language)
    st.write(f"Translated Text: {translated_text}")

# Function to handle hand sign recognition
def hand_sign_to_text(target_language, clf, email_sender):
    cap = cv2.VideoCapture(0)
    SpelledWord = ""
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            st.write("Ignoring empty camera frame.")
            continue

        try:
            SpelledWord = get_prediction(image, clf)
            handle_hand_signs(SpelledWord, target_language, email_sender)

            cv2.putText(image, SpelledWord, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (124, 252, 0), 5, cv2.LINE_AA)
            time.sleep(2)
        except Exception as e:
            st.write(f"Error: {e}")

        cv2.imshow('frame', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press escape to break
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to handle audio input from microphone
def audio_to_text_from_microphone(target_language):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening for live audio input...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"You said: {text}")
            translated_text = translate_text(text, target_language)
            st.write(f"Translated Text: {translated_text}")
            return text
        except sr.UnknownValueError:
            st.write("Google Speech Recognition could not understand the audio.")
            return None
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")
            return None

# Function to process audio file and convert it to text
def audio_to_text_from_file(audio_path, target_language):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        st.write("Processing audio file...")
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"Text extracted from audio: {text}")
            translated_text = translate_text(text, target_language)
            st.write(f"Translated Text: {translated_text}")
            return text
        except sr.UnknownValueError:
            st.write("Google Speech Recognition could not understand the audio.")
            return None
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")
            return None

# Function to extract text from an image
def image_to_text(image, target_language):
    text = pytesseract.image_to_string(image)
    st.write(f"Text extracted from image: {text}")
    translated_text = translate_text(text, target_language)
    st.write(f"Translated Text: {translated_text}")
    return text

# Function to convert text to voice
def text_to_voice(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def extract_text_from_document(file, target_language):
    text = ""
    if file.type == "text/plain":
        text = file.read().decode("utf-8")
    elif file.type == "application/pdf":
        text = extract_text(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    
    st.write("Extracted Text:")
    st.text_area("Original Text", text, height=200)
    
    translated_text = translate_text(text, target_language)
    st.text_area("Translated Text", translated_text, height=200)

    # Download button for translated text
    st.download_button(label="Download Translated Text", data=translated_text, file_name="translated_text.txt", mime="text/plain")

    return translated_text




# Main Streamlit app logic
def main():
    st.set_page_config(page_title="AI-Based Multi-Input Recognition System", page_icon=":guardsman:", layout="wide")

    # Background image for the main page
    def set_background():
        main_bg = "111.jpg"  # Update with your image file in the directory
        main_bg_ext = "jpg"

        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False

    if not st.session_state.user_authenticated:
        st.sidebar.title("Authentication")
        menu = ["Home", "Sign Up", "Login"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Home":
            st.title("Welcome to AI-Based Multi-Input Recognition System")
            st.write("This platform allows you to recognize and translate hand signs, process audio inputs, and extract text from images. Please sign up or log in to continue.")
            set_background()
        
        elif choice == "Sign Up":

            st.subheader("Create New Account")
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type='password')
            set_background()
            if st.button("Sign Up"):
                if new_username and new_password:
                    create_user(new_username, new_password)
                else:
                    st.warning("Please enter both username and password.")
        
        elif choice == "Login":
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            set_background()
            if st.button("Login"):
                if authenticate_user(username, password):
                    st.session_state.user_authenticated = True
                    st.session_state.username = username
                    st.success("Logged in successfully!")
                else:
                    st.error("Invalid username or password.")
    else:
        # If authenticated, go to main app functionality
        st.sidebar.title("Menu")
        language = st.sidebar.selectbox("Select Language", ["Tamil", "English", "Telugu", "Hindi", "Malayalam"])
        task = st.sidebar.selectbox("Choose Task", ["Hand Sign Recognition", "Live Audio Input", "Audio File Processing", "Image Text Extraction", "Document"])

        clf = load_and_train_model()
        email_sender = setup_email()

        set_background()

        if task == "Hand Sign Recognition":
            if st.button("Start Hand Sign Recognition"):
                hand_sign_to_text(language[:2].lower(), clf, email_sender)

        elif task == "Live Audio Input":
            if st.button("Start Live Audio Input"):
                audio_to_text_from_microphone(language[:2].lower())

        elif task == "Audio File Processing":
            uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
            if uploaded_file is not None:
                file_path = os.path.join('temp', uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                audio_to_text_from_file(file_path, language[:2].lower())
            if not os.path.exists('temp'):
                os.makedirs('temp')

        elif task == "Image Text Extraction":
            uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                image_to_text(image, language[:2].lower())

        elif task == "Document":
            document_file = st.file_uploader("Upload a Document (PDF, DOCX, TXT)", type=['pdf', 'docx', 'txt'])
            if document_file:
                translated_text = extract_text_from_document(document_file, language)

if __name__ == "__main__":
    main()