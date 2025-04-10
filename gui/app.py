import streamlit as st
import numpy as np
import pandas as pd
import cv2
import librosa
import sounddevice as sd
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from collections import Counter
from datetime import datetime

# Load models
face_model = load_model('models/fer_cnn_model.h5')
audio_model = load_model('models/audio_emotion_model.h5')

face_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
audio_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Logging setup
CSV_FILE = "results.csv"
try:
    log_df = pd.read_csv(CSV_FILE)
except:
    log_df = pd.DataFrame(columns=["Timestamp", "Face", "Audio", "Final"])

# Predict face emotion
def predict_face_emotion():
    cam = cv2.VideoCapture(0)
    frame = None
    for _ in range(15):  # warmup
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
st.set_page_config(page_title="Real-Time Emotion Detector", layout="centered")
st.title("🎭 Real-Time Emotion Detector (Face + Voice)")

if st.button("▶ Start Emotion Detection"):
    with st.spinner("Capturing face..."):
        face_emotion, frame = predict_face_emotion()
    if frame is not None:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured Frame", channels="RGB")

    with st.spinner("Recording audio..."):
        audio_emotion = predict_audio_emotion()

    st.success("✅ Prediction Complete!")
    st.write(f"🧠 **Face Emotion**: `{face_emotion or 'Not Detected'}`")
    st.write(f"🎧 **Audio Emotion**: `{audio_emotion}`")

    final = Counter([e for e in [face_emotion, audio_emotion] if e is not None]).most_common(1)
    final_emotion = final[0][0] if final else "Unknown"
    st.markdown(f"### 🎯 **Final Emotion**: `{final_emotion}`")

    # Append to log
    log_df.loc[len(log_df)] = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        face_emotion or "None",
        audio_emotion or "None",
        final_emotion
    ]
    log_df.to_csv(CSV_FILE, index=False)

# History section
st.subheader("📜 Prediction History")
st.dataframe(log_df)

st.download_button(
    "📥 Download CSV Log",
    data=log_df.to_csv(index=False),
    file_name="emotion_log.csv",
    mime="text/csv"
)
import matplotlib.pyplot as plt

# Pie Chart for emotion distribution
if not log_df.empty:
    st.subheader("📊 Emotion Distribution")
    fig, ax = plt.subplots()
    log_df['Final'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, shadow=True)
    ax.set_ylabel("")  # Remove y-axis label
    ax.set_title("Final Emotion Breakdown")
    st.pyplot(fig)
    # Emotion Timeline Line Chart
st.subheader("📈 Emotion Timeline")

if not log_df.empty:
    timeline_df = log_df.copy()
    timeline_df["Timestamp"] = pd.to_datetime(timeline_df["Timestamp"])

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(timeline_df["Timestamp"], timeline_df["Final"], marker='o', linestyle='-')
    ax2.set_ylabel("Final Emotion")
    ax2.set_title("Timeline of Final Emotion Predictions")
    plt.xticks(rotation=45)
    st.pyplot(fig2)