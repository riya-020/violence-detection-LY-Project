import cv2
import mediapipe as mp
import numpy as np
import joblib
import time  # for FPS (optional but nice)

# Load trained model
clf = joblib.load("violence_rf_model.pkl")  # keep in same folder

# Init MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


def extract_pose_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if not results.pose_landmarks:
        return None
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(landmarks, dtype=np.float32)


def run_realtime_violence_detection(window_size=30):
    cap = cv2.VideoCapture(0)  # default webcam
    feature_buffer = []
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror

        feats = extract_pose_from_frame(frame)
        if feats is not None:
            feature_buffer.append(feats)
            if len(feature_buffer) > window_size:
                feature_buffer.pop(0)

        label_text = "No prediction"
        color = (0, 255, 255)

        if len(feature_buffer) >= 5:
            video_feat = np.mean(feature_buffer, axis=0).reshape(1, -1)

            if video_feat.shape[1] == clf.n_features_in_:
                proba_all = clf.predict_proba(video_feat)[0]
                prob_non = proba_all[0]
                prob_viol = proba_all[1]

                # 🔥 your favourite part: threshold-based decision
                if prob_viol > 0.55:   # you can tune this: 0.5, 0.55, 0.6...
                    label_text = f"Violence ({prob_viol:.2f})"
                    color = (0, 0, 255)
                    print("⚠️ Violence detected with probability:", prob_viol)
                else:
                    label_text = f"Non-violence ({prob_non:.2f})"
                    color = (0, 255, 0)
            else:
                label_text = "Feature size mismatch!"
                color = (0, 255, 255)

        # Draw pose
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # FPS (optional but cool)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame, label_text, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"FPS: {int(fps)}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, "Press 'q' to quit", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Real-time Violence Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_violence_detection()
