# 🛡️ MaskGuard AI — Face Mask Detection

Real-time face mask detection using your device camera, a CNN `.h5` model, and Streamlit.

---

## 📁 Project Structure

```
face_mask_detector/
├── app.py                          ← Streamlit app
├── requirements.txt                ← Python dependencies
├── face_mask_detection_model.h5    ← ⚠️ YOUR MODEL (place here)
└── README.md
```

---

## 🚀 Quick Start

### 1. Place your model
Copy `face_mask_detection_model.h5` into the same folder as `app.py`.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🎛️ Features

| Feature | Details |
|---|---|
| Live camera | OpenCV captures frames in real-time |
| CNN model | Loads your `.h5` file with Keras |
| Face detection | Haar Cascade (no extra model needed) |
| Green alert | Person wearing a mask ✅ |
| Red alert | Person WITHOUT a mask 🚨 |
| Sidebar controls | Confidence threshold, scale factor, mirroring |
| Stats dashboard | Mask / No-Mask count + compliance % |

---

## 🧠 Model Output Format

The app supports two common output shapes from your `.h5` model:

- **2 outputs** `[mask_prob, no_mask_prob]` — Softmax classifier
- **1 output** `[mask_prob]` — Sigmoid binary classifier (`1 - p` = no-mask)

No code changes needed; the app auto-detects.

---

## ⚙️ Sidebar Settings

| Setting | Default | Description |
|---|---|---|
| Confidence Threshold | 0.60 | Minimum confidence to label a detection |
| Face Scale Factor | 1.10 | Haar cascade scale step |
| Min Neighbors | 5 | Haar cascade quality filter |
| Min Face Size | 50 px | Smallest face to detect |
| Show Confidence Bars | ON | Visual confidence meters |
| Mirror Camera | ON | Flip frame horizontally |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| "Model file not found" | Put `.h5` in same directory as `app.py` |
| "Cannot open camera" | Allow browser/system camera permissions |
| Slow detection | Reduce resolution or use GPU build of TF |
| Wrong predictions | Lower confidence threshold in sidebar |
