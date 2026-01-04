import cv2
import mediapipe as mp
import numpy as np

class HandGestureRecognition:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

    def find_distance(self, p1, p2):
        # Calculate Euclidean distance between two points
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def classify_gesture(self, landmarks):
        # Initialize fingertip landmarks
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_finger_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]

        # Calculating distances for gesture analysis
        thumb_index_distance = self.find_distance(thumb_tip, index_finger_tip)
        thumb_wrist_distance = self.find_distance(thumb_tip, wrist)
        index_middle_distance = self.find_distance(index_finger_tip, middle_finger_tip)

        # Fist Detection
        if all(self.find_distance(finger, wrist) < thumb_wrist_distance for finger in [index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_tip]) and thumb_index_distance < thumb_wrist_distance * 0.5:
            return "Fist"
        
        # Thumbs Up Detection
        elif thumb_index_distance < thumb_wrist_distance and all(finger.y > thumb_tip.y for finger in [index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_tip]):
            return "Thumbs Up"
        
        elif thumb_tip.y > wrist.y and all(finger.y < thumb_tip.y for finger in [index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_tip]):
          return "Thumbs Down"
        
        elif thumb_index_distance < thumb_wrist_distance * 0.4:
            return "OK Sign"

        # Peace Sign Detection
        elif index_middle_distance > thumb_wrist_distance * 0.5 and all(finger.y < wrist.y for finger in [index_finger_tip, middle_finger_tip]) and all(finger.y < wrist.y for finger in [ring_finger_tip, pinky_tip]):
            return "Peace Sign"

        elif all(self.find_distance(wrist, finger) > thumb_wrist_distance * 0.75 for finger in [index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_tip]):
            return "Open Hand"
        
        else:
            return "Neutral"

    def detect_and_classify_hand_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        gesture = "Neutral"  # Default gesture
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture = self.classify_gesture(hand_landmarks.landmark)
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = self.detect_and_classify_hand_gesture(frame)
            cv2.imshow('Hand Gesture Recognition', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()

if __name__ == '__main__':
    recognizer = HandGestureRecognition()
    recognizer.run()
