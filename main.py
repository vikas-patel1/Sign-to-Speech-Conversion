import cv2
import numpy as np
import pickle
import os

from hand_tracker import MyHands
from features import compute_distance_matrix
from matcher import match_gesture


# ---------------- CONFIG ---------------- #

HEIGHT, WIDTH = 360, 640
KEYPOINTS = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]
SAMPLES_PER_GESTURE = 30
TOLERANCE = 15

# ---------------- PATH SETUP ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_DIR = os.path.join(BASE_DIR, "gesture_files")

os.makedirs(GESTURE_DIR, exist_ok=True)

# ---------------- MODE ---------------- #

mode = int(input("Enter 1 to train, enter 0 to recognize: "))

# ---------------- CAMERA ---------------- #

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)

hand_tracker = MyHands()

known_gestures = []
gesture_names = []
train_idx = 0

# ---------------- LOAD / TRAIN FILE ---------------- #


if mode == 1:
    n = int(input("How many gestures? "))
    for i in range(n):
        gesture_names.append(input(f"Name for gesture #{i+1}: "))

    file_name = input("Training file name (default): ").strip()
    if file_name == "":
        file_name = "default"

else:
    file_name = input("Training file to load (default): ").strip()
    if file_name == "":
        file_name = "default"


file_path = os.path.join(GESTURE_DIR, file_name + ".pkl")


if mode == 0:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No training file found at {file_path}")

    with open(file_path, "rb") as f:
        gesture_names = pickle.load(f)
        known_gestures = pickle.load(f)

# ---------------- LOOP ---------------- #

while True:
    ret, frame = cam.read()
    if not ret:
        break

    hands = hand_tracker.get_landmarks(frame)

    # ---------- TRAIN ---------- #
    if mode == 1 and hands:
        print(f"Show '{gesture_names[train_idx]}', press 't' and hold")

        if cv2.waitKey(1) & 0xFF == ord('t'):
            samples = []
            print("Capturing...")

            while len(samples) < SAMPLES_PER_GESTURE:
                ret, frame = cam.read()
                if not ret:
                    continue

                hands = hand_tracker.get_landmarks(frame)
                if hands:
                    samples.append(compute_distance_matrix(hands[0]))

                cv2.imshow("Gesture Trainer", frame)
                cv2.waitKey(1)

            known_gestures.append(np.mean(samples, axis=0))
            train_idx += 1

            if train_idx == len(gesture_names):
                with open(file_path, "wb") as f:
                    pickle.dump(gesture_names, f)
                    pickle.dump(known_gestures, f)
                print("Training completed")
                mode = 0

    # ---------- RECOGNIZE ---------- #
    if mode == 0 and hands:
        unknown = compute_distance_matrix(hands[0])
        label = match_gesture(
            unknown, known_gestures, gesture_names, KEYPOINTS, TOLERANCE
        )

        cv2.putText(
            frame, label, (100, 175),
            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 4
        )

    for hand in hands:
        for i in KEYPOINTS:
            cv2.circle(frame, hand[i], 15, (255, 0, 255), 2)

    cv2.imshow("Gesture Trainer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
