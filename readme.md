# 📰🤖 AI Media Authenticity Project

Detect **Fake News (Text)** and **Deepfake Faces (Images)** using AI & Machine Learning.

---

## 📌 Project Overview
This project demonstrates how AI/ML can be applied to **media authenticity**:
- 📰 **Fake News Detection** → Classify news articles as *REAL* or *FAKE*.
- 🖼 **Deepfake Detection** → Detect whether a face image is *REAL* or *FAKE*.

It includes:
- Data preprocessing utilities
- Training pipelines for both tasks
- A simple Flask web app to test text or upload media

---

## 📂 Project Structure
```
ai_media_authenticity/
├── README.md
├── requirements.txt
├── data/
│   ├── fake_news/         # Kaggle fake/real news dataset (train.csv/test.csv)
│   └── deepfake/          # Kaggle deepfake face dataset (real/ /fake/)
├── fake_news/
│   ├── train_fake.py      # train fake news classifier
│   ├── infer_fake.py      # inference script for news
│   └── vectorizer.pkl
├── deepfake/
│   ├── train_deepfake.py  # train CNN on face images
│   ├── infer_deepfake.py  # inference script for faces
│   └── models/
├── app/
│   ├── app.py             # Flask web app
│   ├── templates/
│   │   ├── index.html
│   │   └── results.html
│   └── static/
└── utils/
    ├── preprocessing.py
    └── video_utils.py
```

---

## 📊 Datasets Used
1. **Fake News Dataset**  
   - Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
   - Contains news articles labeled as **FAKE** or **REAL**.  
   - We merged `Fake.csv` + `True.csv` → `train.csv` with columns:  
     ```
     title,text,label
     "Breaking News...","News body here","FAKE"
     "Government...","News body here","REAL"
     ```

2. **Deepfake Faces Dataset**  
   - Kaggle: [Real and Fake Face Dataset](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)  
   - ~70,000 images, balanced between **REAL** and **FAKE** faces.  
   - Used for CNN-based image classification.  

---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mahek-08/fake-news-deepfake.git
   cd fake-news-deepfake
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Train Fake News Classifier
```bash
python -m fake_news.train_fake
```

### 2. Train Deepfake Detector
```bash
python -m deepfake.train_deepfake
```

### 3. Run Flask App
```bash
python -m app.app
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## 🧠 Models Used
- **Fake News (Text)** → Logistic Regression with TF-IDF vectorization.  
- **Deepfake Faces (Images)** → CNN built with TensorFlow/Keras (Conv2D, MaxPooling, Dense).  

---

## 📸 Screenshots
### 🔹 Web App Home
![App Home](screenshots/ui_home.png)

### 🔹 Fake News Detection
![Fake News Example](screenshots/result_fake.png)

### 🔹 Deepfake Detection
![Deepfake Example](screenshots/result_real.png)  

---

## ✨ Future Improvements
- Replace Logistic Regression with **BERT/RoBERTa** for news.  
- Use **EfficientNet / ResNet** for deepfake images.  
- Deploy via **Streamlit / Hugging Face Spaces** for easy online access.  

---

## 👩‍💻 Author
**Mahek** – AI/ML Enthusiast