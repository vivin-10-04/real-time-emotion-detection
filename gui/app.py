import streamlit as st
import numpy as np
import pandas as pd
import cv2
import librosa
import sounddevice as sd
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from datetime import datetime
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pyttsx3
import random

# Load models
face_model = load_model('models/fer_cnn_model.h5')
audio_model = load_model('models/audio_emotion_model.h5')

face_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
audio_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Load song dataset
try:
    songs_df = pd.read_csv("data/bollywood_songs.csv")
except:
    songs_df = pd.DataFrame(columns=["song_name", "artist_name", "spotify_track_link", "thumbnail_link", "Sad Ballad", "Happy", "Party Anthem", "Motivational", "Romantic Ballad"])

# Log file
CSV_FILE = "results.csv"
try:
    log_df = pd.read_csv(CSV_FILE)
except:
    log_df = pd.DataFrame(columns=["Timestamp", "Face", "Audio", "Final"])

# Emotion to genre mapping
emotion_genre_map = {
    "happy": ["Happy", "Party Anthem", "Motivational"],
    "sad": ["Sad Ballad"],
    "angry": ["Motivational", "Rock Fusion"],
    "neutral": ["Romantic Ballad"],
    "surprised": ["Party Anthem"],
    "fearful": ["Motivational"],
    "disgust": ["Lyrical Rap"]
}

# Track already shown songs to avoid repeats
if "shown_songs" not in st.session_state:
    st.session_state.shown_songs = defaultdict(set)

# TTS
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        st.warning("🗣️ Text-to-speech failed.")

# Face emotion
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
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return None, frame

    x, y, w, h = faces[0]
    roi = gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, (48, 48))
    roi = img_to_array(roi) / 255.0
    roi = np.expand_dims(np.expand_dims(roi, axis=0), axis=-1)

    pred = face_model.predict(roi)[0]
    return face_labels[np.argmax(pred)], frame

# Audio emotion
def predict_audio_emotion():
    audio = sd.rec(int(3 * 22050), samplerate=22050, channels=1)
    sd.wait()
    mfcc = librosa.feature.mfcc(y=audio.flatten(), sr=22050, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    pred = audio_model.predict(np.expand_dims(mfcc_scaled, axis=0))[0]
    return audio_labels[np.argmax(pred)]

# App UI
st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("🎭 Real-Time Emotion Detector")

if st.button("▶ Start Emotion Detection"):
    with st.spinner("Analyzing face..."):
        face_emotion, frame = predict_face_emotion()
    if frame is not None:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured Face")

    with st.spinner("Analyzing voice..."):
        audio_emotion = predict_audio_emotion()

    st.success("Detection complete!")
    st.write(f"🧠 Face: `{face_emotion or 'None'}`")
    st.write(f"🎧 Audio: `{audio_emotion or 'None'}`")

    final = Counter([e for e in [face_emotion, audio_emotion] if e]).most_common(1)
    final_emotion = final[0][0] if final else "Unknown"
    st.markdown(f"### 🎯 Final Emotion: `{final_emotion}`")

    # Save log
    log_df.loc[len(log_df)] = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), face_emotion or "None", audio_emotion or "None", final_emotion]
    log_df.to_csv(CSV_FILE, index=False)

    # 🎶 Suggest a song
    st.subheader("🎶 Suggested Bollywood Song")
    genres = emotion_genre_map.get(final_emotion.lower(), [])
    filtered = songs_df.copy()
    for genre in genres:
        if genre in filtered.columns:
            filtered = filtered[filtered[genre] == 1]

    shown = st.session_state.shown_songs[final_emotion]
    remaining = filtered[~filtered["song_name"].isin(shown)]

    if not remaining.empty:
        song = remaining.sample(1).iloc[0]
        st.session_state.shown_songs[final_emotion].add(song["song_name"])

        st.write(f"**{song['song_name']}** by *{song['artist_name']}*")
        speak(f"You seem {final_emotion}. Here's a song for you: {song['song_name']} by {song['artist_name']}.")

        if pd.notna(song.get("thumbnail_link")):
            st.image(song["thumbnail_link"], width=300)

        link = song.get("spotify_track_link", "")
        if "spotify.com" in link:
            embed = link.replace("open.spotify.com/track", "open.spotify.com/embed/track")
            st.markdown(f'<iframe src="{embed}" width="100%" height="80" frameborder="0" allowtransparency="true" allow="autoplay"></iframe>', unsafe_allow_html=True)
        else:
            st.markdown(f"[▶️ Listen Here]({link})")
    else:
        st.info("No more new songs left for this emotion!")

# Logs & Charts
st.subheader("📜 Prediction History")
st.dataframe(log_df)
st.download_button("📥 Download CSV", log_df.to_csv(index=False), "emotion_log.csv")

if not log_df.empty:
    st.subheader("📊 Emotion Distribution")
    fig, ax = plt.subplots()
    log_df["Final"].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

    st.subheader("📈 Emotion Over Time")
    fig2, ax2 = plt.subplots()
    timeline = log_df.copy()
    timeline["Timestamp"] = pd.to_datetime(timeline["Timestamp"])
    ax2.plot(timeline["Timestamp"], timeline["Final"], marker='o')
    ax2.set_ylabel("Emotion")
    ax2.set_xlabel("Time")
    plt.xticks(rotation=45)
    st.pyplot(fig2)