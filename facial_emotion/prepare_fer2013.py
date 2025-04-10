import os
import pandas as pd
import numpy as np
import cv2

# Paths
csv_path = 'fer2013.csv'
output_dir = 'data/fer2013'
emotion_map = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy",
    4: "sad", 5: "surprise", 6: "neutral"
}

# Create output directories
for split in ['Training', 'PublicTest']:
    for emotion in emotion_map.values():
        path = os.path.join(output_dir, split, emotion)
        os.makedirs(path, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

print(f"Total samples: {len(df)}")

# Convert CSV to image files
for i, row in df.iterrows():
    emotion = emotion_map[row['emotion']]
    usage = 'Training' if row['Usage'] == 'Training' else 'PublicTest'
    pixels = np.array(row['pixels'].split(), dtype='uint8').reshape(48, 48)

    img_path = os.path.join(output_dir, usage, emotion, f"{i}.jpg")
    cv2.imwrite(img_path, pixels)

print("✅ Dataset successfully created at: data/fer2013")