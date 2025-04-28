import os
import librosa
import numpy as np

DATA_PATH = 'data/ravdess_audio'

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f" Error loading {file_path}: {e}")
        return None

features = []
labels = []

print("🔍 Checking RAVDESS audio files...")

for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.wav'):
            emotion_code = file.split('-')[2]
            emotion = emotion_map.get(emotion_code)
            if emotion:
                path = os.path.join(root, file)
                mfcc = extract_features(path)
                if mfcc is not None:
                    features.append(mfcc)
                    labels.append(emotion)

print(f"\n Total files loaded: {len(features)}")
print(f"🎭 Emotion labels found: {set(labels)}")
