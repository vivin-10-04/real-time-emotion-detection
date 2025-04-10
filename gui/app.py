import streamlit as st
import numpy as np
import pandas as pd
import cv2
import librosa
import sounddevice as sd
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import pyttsx3
import random

# Load models
face_model = load_model('models/fer_cnn_model.h5')
audio_model = load_model('models/audio_emotion_model.h5')

face_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
audio_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Load Bollywood song dataset
try:
    songs_df = pd.read_csv("data/bollywood_songs.csv")
except:
    songs_df = pd.DataFrame(columns=["song_name", "artist_name", "spotify_track_link", "thumbnail_link"])

# Emotion to sample song names (used for matching)
emotion_map = {
    "happy": ["Ilahi", "Zinda", "Gallan Goodiyan"],
    "sad": ["Channa Mereya", "Agar Tum Saath Ho", "Tujhe Bhula Diya"],
    "angry": ["Sadda Haq", "Ziddi Dil"],
    "neutral": ["Kabira", "Phir Se Ud Chala"],
    "surprised": ["Oh Gujariya", "Dil Dhadakne Do"],
    "fearful": ["Bhaag Milkha Bhaag"],
    "disgust": ["Emotional Atyachar"]
}

# Initialize session state for unique song per emotion
if "last_song" not in st.session_state:
    st.session_state.last_song = {}

CSV_FILE = "results.csv"
try:
    log_df = pd.read_csv(CSV_FILE)
except:
    log_df = pd.DataFrame(columns=["Timestamp", "Face", "Audio", "Final"])

# TTS function
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        st.warning("🗣️ Text-to-speech failed (pyttsx3 issue).")

# Predict face emotion
def predict_face_emotion():
    cam = cv2.VideoCapture(0)
    frame = None
    for _ in range(10):
        ret, temp = cam.read()
        if ret:
            frame = temp
        cv2.waitKey(100)
    cam.release()

    if frame is None:
        return None, None

    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return None, frame

    x, y, w, h = faces[0]
    roi = gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, (48, 48))
    roi = img_to_array(roi) / 255.0
    roi = np.expand_dims(roi, axis=0)
    roi = np.expand_dims(roi, axis=-1)

    prediction = face_model.predict(roi)[0]
    return face_labels[np.argmax(prediction)], frame

# Predict audio emotion
def predict_audio_emotion():
    duration = 3
    sr = 22050
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    mfcc = librosa.feature.mfcc(y=audio.flatten(), sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    mfcc_scaled = np.expand_dims(mfcc_scaled, axis=0)

    prediction = audio_model.predict(mfcc_scaled)[0]
    return audio_labels[np.argmax(prediction)]

# Streamlit UI
st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("🎭 Real-Time Emotion Detector (Face + Voice)")

if st.button("▶ Start Emotion Detection"):
    with st.spinner("Analyzing face..."):
        face_emotion, frame = predict_face_emotion()
    if frame is not None:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured Face")

    with st.spinner("Analyzing voice..."):
        audio_emotion = predict_audio_emotion()

    st.success("Detection complete!")
    st.write(f"🧠 **Face Emotion**: `{face_emotion or 'Not Detected'}`")
    st.write(f"🎧 **Audio Emotion**: `{audio_emotion or 'Not Detected'}`")

    final = Counter([e for e in [face_emotion, audio_emotion] if e]).most_common(1)
    final_emotion = final[0][0] if final else "Unknown"
    st.markdown(f"### 🎯 Final Emotion: `{final_emotion}`")

    # Log to CSV
    log_df.loc[len(log_df)] = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        face_emotion or "None",
        audio_emotion or "None",
        final_emotion
    ]
    log_df.to_csv(CSV_FILE, index=False)

    # 🎶 Suggested Bollywood Song
    st.subheader("🎶 Suggested Bollywood Song")
    titles = emotion_map.get(final_emotion.lower(), [])
    matched = songs_df[songs_df["song_name"].isin(titles)]

    if not matched.empty:
        # Filter out last shown song
        last = st.session_state.last_song.get(final_emotion)
        options = matched[~matched["song_name"].isin([last])] if last else matched
        if options.empty:  # if all are shown, allow all again
            options = matched
        song = options.sample(1).iloc[0]

        # Save the song shown for this emotion
        st.session_state.last_song[final_emotion] = song["song_name"]

        track = song["song_name"]
        artist = song["artist_name"]
        link = song.get("spotify_track_link", "")
        image = song.get("thumbnail_link", "")

        st.write(f"**{track}** by *{artist}*")
        speak(f"You seem {final_emotion}. Here's a song for you: {track} by {artist}.")

        if pd.notna(image):
            st.image(image, width=300)

        if "youtube.com" in link or "spotify.com" in link:
            if "v=" in link:
                video_id = link.split("v=")[-1].split("&")[0]
                st.video(f"https://www.youtube.com/embed/{video_id}")
            else:
                st.markdown(f"[▶️ Listen Here]({link})")
        else:
            st.markdown(f"[▶️ Listen Here]({link})")
    else:
        st.info("No matching song found for this emotion.")

# Logs
st.subheader("📜 Prediction History")
st.dataframe(log_df)
st.download_button("📥 Download CSV", log_df.to_csv(index=False), "emotion_log.csv")

# Charts
if not log_df.empty:
    st.subheader("📊 Emotion Distribution")
    fig, ax = plt.subplots()
    log_df["Final"].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

    st.subheader("📈 Emotion Over Time")
    timeline = log_df.copy()
    timeline["Timestamp"] = pd.to_datetime(timeline["Timestamp"])
    fig2, ax2 = plt.subplots()
    ax2.plot(timeline["Timestamp"], timeline["Final"], marker='o')
    ax2.set_ylabel("Final Emotion")
    ax2.set_xlabel("Time")
    plt.xticks(rotation=45)
    st.pyplot(fig2)