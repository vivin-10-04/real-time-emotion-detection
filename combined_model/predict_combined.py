import numpy as np
import cv2
import librosa
import sounddevice as sd
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from collections import Counter

# Load models
face_model = load_model('models/fer_cnn_model.h5')
audio_model = load_model('models/audio_emotion_model.h5')

# Emotion label mappings
face_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
audio_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Predict emotion from facial image
def predict_face_emotion(frame):
    if frame is None:
        print("❌ No frame captured from camera.")
        return None

    # Brightness correction (optional)
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    print(f"🔍 Faces detected: {len(faces)}")

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    roi = gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, (48, 48))
    roi = img_to_array(roi) / 255.0
    roi = np.expand_dims(roi, axis=0)
    roi = np.expand_dims(roi, axis=-1)

    prediction = face_model.predict(roi)[0]
    predicted_label = face_labels[np.argmax(prediction)]
    return predicted_label

# Predict emotion from audio recording
def predict_audio_emotion():
    duration = 3  # seconds
    sr = 22050

    print("🎙️ Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    print("✅ Done recording.")

    mfccs = librosa.feature.mfcc(y=audio.flatten(), sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    mfccs_scaled = np.expand_dims(mfccs_scaled, axis=0)

    prediction = audio_model.predict(mfccs_scaled)[0]
    predicted_label = audio_labels[np.argmax(prediction)]
    return predicted_label

# Combine predictions from both face and voice
def predict_combined():
    cam = cv2.VideoCapture(0)

    # Warm up the camera
    frame = None
    for i in range(15):
        ret, temp = cam.read()
        if ret:
            frame = temp
        cv2.waitKey(100)

    cam.release()

    if frame is not None:
        cv2.imshow("📷 Captured Frame", frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    face_emotion = predict_face_emotion(frame)
    audio_emotion = predict_audio_emotion()

    print(f"🧠 Face Emotion: {face_emotion}")
    print(f"🎧 Audio Emotion: {audio_emotion}")

    # Combine (ignore None)
    combined = [e for e in [face_emotion, audio_emotion] if e is not None]
    final = Counter(combined).most_common(1)
    final_emotion = final[0][0] if final else None

    print(f"🎯 Final Emotion: {final_emotion}")

# Run
if __name__ == "__main__":
    predict_combined()