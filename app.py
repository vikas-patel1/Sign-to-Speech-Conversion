import streamlit as st
import cv2
import numpy as np
import pickle
import time

from hand_tracker import MyHands
from features import compute_distance_matrix
from matcher import match_gesture

# ---------------- CONFIG ---------------- #

HEIGHT, WIDTH = 360, 640
KEYPOINTS = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]
SAMPLES_PER_GESTURE = 30
TOLERANCE = 15

# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")

st.title("🤖 Hand Gesture Recognition using Computer Vision")

st.sidebar.header("Controls")

mode = st.sidebar.radio(
    "Select Mode",
    ["Recognize Gesture", "Train New Gestures"]
)

file_name = st.sidebar.text_input(
    "Training file name",
    value="default"
) + ".pkl"

start = st.sidebar.button("Start Camera")
stop = st.sidebar.button("Stop Camera")

FRAME_WINDOW = st.image([])

status = st.empty()

# ---------------- SESSION STATE ---------------- #

if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True

if stop:
    st.session_state.run = False

# ---------------- CAMERA SETUP ---------------- #

hand_tracker = MyHands()
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)

# ---------------- LOAD TRAINED DATA ---------------- #

known_gestures = []
gesture_names = []

if mode == "Recognize Gesture":
    try:
        with open(file_name, "rb") as f:
            gesture_names = pickle.load(f)
            known_gestures = pickle.load(f)
        status.success("Training data loaded")
    except:
        status.error("Training file not found")

# ---------------- TRAINING INPUT ---------------- #

if mode == "Train New Gestures":
    n = st.sidebar.number_input(
        "Number of gestures",
        min_value=1,
        max_value=10,
        step=1
    )

    gesture_names = [
        st.sidebar.text_input(f"Gesture {i+1} name")
        for i in range(n)
    ]

    capture = st.sidebar.button("Capture Gesture")

# ---------------- MAIN LOOP ---------------- #

while st.session_state.run:
    ret, frame = cam.read()
    if not ret:
        break

    hands = hand_tracker.get_landmarks(frame)

    # ---------- TRAIN MODE ---------- #
    if mode == "Train New Gestures" and hands and capture:
        samples = []
        status.info("Capturing gesture samples...")

        while len(samples) < SAMPLES_PER_GESTURE:
            ret, frame = cam.read()
            hands = hand_tracker.get_landmarks(frame)

            if hands:
                samples.append(compute_distance_matrix(hands[0]))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)

        known_gestures.append(np.mean(samples, axis=0))

        if len(known_gestures) == len(gesture_names):
            with open(file_name, "wb") as f:
                pickle.dump(gesture_names, f)
                pickle.dump(known_gestures, f)
            status.success("Training completed")

    # ---------- RECOGNITION MODE ---------- #
    if mode == "Recognize Gesture" and hands:
        unknown = compute_distance_matrix(hands[0])
        label = match_gesture(
            unknown,
            known_gestures,
            gesture_names,
            KEYPOINTS,
            TOLERANCE
        )

        cv2.putText(
            frame, label, (80, 120),
            cv2.FONT_HERSHEY_COMPLEX, 2,
            (255, 0, 0), 3
        )

    # ---------- DRAW LANDMARKS ---------- #
    for hand in hands:
        for i in KEYPOINTS:
            cv2.circle(frame, hand[i], 8, (255, 0, 255), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)

    time.sleep(0.02)

cam.release()
