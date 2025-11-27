# Violence Detection using MediaPipe Pose

This project uses **MediaPipe Pose** + a **RandomForest classifier** to classify a scene as:

- `0` → Non-violence  
- `1` → Violence  

The model is trained offline in Google Colab and used in real-time on a local machine.

---

## Files

- `realtime_violence.py`  
  Uses your webcam + MediaPipe Pose + the trained model to show **real-time violence / non-violence prediction** with confidence.

- `violence_rf_model.pkl`  
  Trained RandomForest model (no need to retrain).  
  Any system can load this file and directly do predictions.

---

## How the model works (high-level)

- Uses **MediaPipe Pose** to get 33 body landmarks.
- For each landmark, we take `[x, y, z, visibility]` → 4 values.
- That gives `33 × 4 = 132` features per frame.
- For a short sequence of frames, we **average** the landmark features across frames.
- Final 132D feature vector is passed to the RandomForest model.
- Model outputs probabilities for:
  - Non-violence
  - Violence

---

## Running the real-time demo (local)

Requirements (install once):

```bash
pip install opencv-python mediapipe numpy scikit-learn joblib

Then run:

```bash
python realtime_violence.py
A webcam window will open.

It draws the pose skeleton.

At the top it shows:

Non-violence (p) or Violence (p) where p is model confidence.

Press q to quit.
Using the trained model in another project (e.g., website backend)

In Python:

import numpy as np
import joblib

# Load model (make sure path is correct)
clf = joblib.load("violence_rf_model.pkl")

def predict_violence_from_features(features_1d):
    """
    features_1d: 1D array-like of length 132
    (same format as training features)
    """
    x = np.array(features_1d).reshape(1, -1)
    proba = clf.predict_proba(x)[0]  # [P(non-violence), P(violence)]
    prob_non = float(proba[0])
    prob_viol = float(proba[1])
    label = int(np.argmax(proba))  # 0 or 1
    return prob_non, prob_viol, label

You can integrate this into an API and call it from a website.

