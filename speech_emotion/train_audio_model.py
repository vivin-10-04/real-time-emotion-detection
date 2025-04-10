import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

print("🔥 Script started...")

# Path to RAVDESS dataset
DATA_PATH = 'data/ravdess_audio'

# Emotion codes to labels
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
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=3.0)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

def load_data():
    print("🔍 Loading audio data...")
    features = []
    labels = []

    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith('.wav'):
                emotion_code = file.split('-')[2]
                label = emotion_map.get(emotion_code)
                if label:
                    path = os.path.join(root, file)
                    feature = extract_features(path)
                    if feature is not None:
                        features.append(feature)
                        labels.append(label)

    return np.array(features), np.array(labels)

def train_model():
    X, y = load_data()
    print(f"✅ Loaded {len(X)} samples with shape: {X.shape}")

    if X.shape[0] == 0:
        print("⚠️ No data loaded. Check your dataset path.")
        return

    # Encode labels
    le = LabelEncoder()
    y_encoded = to_categorical(le.fit_transform(y))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Build model
    model = Sequential([
        Dense(256, activation='relu', input_shape=(40,)),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(y_encoded.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("🚀 Starting training...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/audio_emotion_model.h5'
    model.save(model_path)
    print(f"🎉 Model saved successfully at: {model_path}")

if __name__ == "__main__":
    print("🎧 Starting audio emotion model training...")
    train_model()