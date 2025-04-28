<h1 align="center">🎭MoodSync: Real-time Emotion Detection and Song Recommendation</h1>
<p align="center">
  👁️ Facial + 🎙️ Voice Emotion Recognition using Deep Learning & Streamlit
</p>

---

## 🔍 What It Does

- 👀 Detects **facial emotions** using webcam and CNN (FER2013)  
- 🎧 Detects **voice emotions** using mic and MFCC + DNN (RAVDESS)  
- 🎯 Combines both predictions to show a **final emotion**  
- 🎶 Suggests a **Bollywood song** with album art and playable link  
- 🔊 Speaks the emotion and song using **Text-to-Speech**  
- 📊 Shows emotion **history**, pie chart, and timeline  
- 🧪 All wrapped inside a user-friendly **Streamlit GUI**  
- 📦 Logs every prediction to a downloadable `results.csv`  

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/jatinverma2703/real-time-emotion-detection.git
cd real-time-emotion-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run gui/app.py
```

> ⚠️ Make sure your webcam and mic are accessible.

---

## 🧠 Tech Stack

- **Frontend**: Streamlit  
- **Face Model**: CNN trained on FER2013  
- **Audio Model**: DNN trained on RAVDESS  
- **Libraries**:
  - OpenCV  
  - TensorFlow / Keras  
  - LibROSA + SoundDevice  
  - Pyttsx3  
  - Pandas + Matplotlib  

---

## 📸 Screenshots

| GUI Interface | Emotion Graphs |
|---------------|----------------|
| *(Insert GUI screenshot or GIF here)* | *(Insert pie + timeline chart here)* |

---

## 📦 Model Info

| Model File               | Dataset   | Purpose                    |
|--------------------------|-----------|----------------------------|
| `fer_cnn_model.h5`       | FER2013   | Facial emotion detection   |
| `audio_emotion_model.h5` | RAVDESS   | Audio emotion classification |

---

## 📥 Output Log

- All predictions are saved to `results.csv`  
- Format includes: `Timestamp`, `Face`, `Audio`, `Final`  
- Can be downloaded from GUI  

---

## 📊 Visualization

- Pie chart showing emotion distribution  
- Timeline showing emotion over time  
- Interactive data table of results  

---

## 📁 Dataset Sources

- [FER2013 Facial Emotion Dataset (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)  
- [RAVDESS Emotional Speech Audio (Kaggle)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)  
- [Bollywood Songs Dataset (Kaggle)](https://www.kaggle.com/)  

---

## 👨‍💻 Authors

- [Jatin Verma](https://github.com/jatinverma2703)  
- [Vivin Pathak](https://github.com/vivin-10-04)

---

## 💡 Future Ideas

- 🌐 Deploy to Streamlit Cloud  
- 🖥️ Add continuous webcam preview  
- 📱 Build mobile-friendly version  
- 🎼 Auto-tag songs using valence scores  
- 🔐 Add Spotify login + playlist export  

---

> Made with ❤️ using Machine Learning, Audio Vision, and Bollywood 🎶
