import cv2
import mediapipe as mp


class MyHands:
    def __init__(self, max_hands=1, det_conf=0.5, track_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )

    def get_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        results = self.hands.process(frame_rgb)

        all_hands = []
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                hand = []
                for lm in hand_lms.landmark:
                    cx = int(lm.x * width)
                    cy = int(lm.y * height)
                    hand.append((cx, cy))
                all_hands.append(hand)

        return all_hands
