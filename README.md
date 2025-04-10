<h1 align="center">🎭 Real-Time Emotion Detection</h1>
<p align="center">
  👁️ Facial + 🎙️ Voice Emotion Recognition using Deep Learning & Streamlit
</p>

---

## 🔍 What It Does

- 👀 Detects **facial emotions** using webcam and CNN (FER2013)  
- 🎧 Detects **voice emotions** using mic and MFCC + DNN (RAVDESS)  
- 🎯 Combines both predictions to show a **final emotion**  
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

---

## 🧠 Tech Stack

- **Frontend**: Streamlit  
- **Face Model**: CNN trained on FER2013  
- **Audio Model**: DNN trained on RAVDESS  
- **Libraries**: OpenCV, Librosa, SoundDevice, TensorFlow/Keras, Pandas, Matplotlib  

---

## 📸 Screenshots

| GUI | Chart View |
|-----|------------|
| *(Add your screenshots here)* | *(or a GIF preview)* |

---

## 📦 Model Info

| Model File               | Dataset   | Purpose                    |
|--------------------------|-----------|----------------------------|
| `fer_cnn_model.h5`       | FER2013   | Facial emotion detection   |
| `audio_emotion_model.h5` | RAVDESS   | Audio emotion classification |

---

## 📥 Output Log

- All results are saved to `results.csv`  
- Format includes: `Timestamp`, `Face`, `Audio`, `Final`  
- Download CSV directly from the Streamlit GUI  

---

## 📊 Visualization

- Pie chart of final emotion distribution  
- Timeline chart of emotional trends over time  
- Live data table with every prediction  

---

## 📁 Dataset Sources

- [FER2013 Facial Emotion Dataset (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)  
- [RAVDESS Emotional Speech Audio (Kaggle)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)  

---

## 👨‍💻 Author

Made with ❤️ by [Jatin Verma](https://github.com/jatinverma2703)

---

## ⭐ Future Features

- 🌐 Deploy to Streamlit Cloud  
- 🖥️ Add continuous webcam preview  
- 📱 Build mobile-friendly version  
- 🧠 Use attention-based models for fusion  
