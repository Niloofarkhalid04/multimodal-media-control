import cv2
import mediapipe as mp
import pyautogui
import time
import math
from collections import deque
from cvzone.HandTrackingModule import HandDetector

# ---------------- INIT ----------------
cap = cv2.VideoCapture(0)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

hand_detector = HandDetector(maxHands=1, detectionCon=0.7)

cooldown_time = 2.5
last_action_time = 0

status_text = "IDLE"

# Head control state
head_state = "NEUTRAL"   # NEUTRAL, LEFT, RIGHT

# Blink detection
blink_history = deque(maxlen=3)

# ---------------- UTILS ----------------
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(eye_points):
    A = distance(eye_points[1], eye_points[5])
    B = distance(eye_points[2], eye_points[4])
    C = distance(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

# ---------------- LOOP ----------------
while True:
    success, img = cap.read()
    if not success:
        break

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(img_rgb)
    hands, img = hand_detector.findHands(img)
    current_time = time.time()

    # ================= FACE + EYES =================
    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0]

        # -------- Head turn (nose) --------
        nose = face.landmark[1]
        nose_x = int(nose.x * w)
        face_center_x = w // 2

        offset = nose_x - face_center_x

        # Visual guides
        cv2.line(img, (face_center_x, 0), (face_center_x, h), (255, 0, 0), 1)
        cv2.circle(img, (nose_x, int(nose.y * h)), 4, (0, 255, 0), -1)

        # -------- Blink detection (TOP PRIORITY) --------
        left_eye_ids = [33, 160, 158, 133, 153, 144]
        left_eye = [(int(face.landmark[i].x * w),
                     int(face.landmark[i].y * h)) for i in left_eye_ids]

        ear = eye_aspect_ratio(left_eye)
        blink_history.append(ear)

        if len(blink_history) == 3:
            avg_ear = sum(blink_history) / 3

            if avg_ear < 0.18 and current_time - last_action_time > cooldown_time:
                pyautogui.press("playpause")
                status_text = "PLAY / PAUSE (BLINK)"
                last_action_time = current_time
                blink_history.clear()
                head_state = "LOCKED"  # temporarily lock head

        # -------- Head neutral reset --------
        if abs(offset) < 30:
            head_state = "NEUTRAL"

        # -------- Head LEFT / RIGHT --------
        if current_time - last_action_time > cooldown_time:

            # RIGHT → NEXT
            if offset > 60 and head_state == "NEUTRAL":
                pyautogui.press("nexttrack")
                status_text = "NEXT TRACK (HEAD RIGHT)"
                last_action_time = current_time
                head_state = "RIGHT"

            # LEFT → PREVIOUS
            elif offset < -60 and head_state == "NEUTRAL":
                pyautogui.press("prevtrack")
                status_text = "PREVIOUS TRACK (HEAD LEFT)"
                last_action_time = current_time
                head_state = "LEFT"

    # ================= HAND =================
    if hands and current_time - last_action_time > cooldown_time:
        hand = hands[0]
        fingers = hand_detector.fingersUp(hand)

        if fingers == [0, 1, 0, 0, 0]:
            pyautogui.press("volumeup")
            status_text = "VOLUME UP (HAND)"
            last_action_time = current_time

        elif fingers == [0, 1, 1, 1, 0]:
            pyautogui.press("volumedown")
            status_text = "VOLUME DOWN (HAND)"
            last_action_time = current_time

        elif fingers == [0, 0, 0, 0, 0]:
            pyautogui.press("volumemute")
            status_text = "MUTE (HAND)"
            last_action_time = current_time

        elif fingers == [1, 1, 1, 1, 1]:
            pyautogui.press("volumemute")
            status_text = "UNMUTE (HAND)"
            last_action_time = current_time

    # ================= UI =================
    cv2.rectangle(img, (20, 20), (600, 100), (40, 40, 40), -1)
    cv2.putText(img, "MULTIMODAL MEDIA CONTROL",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(img, f"STATUS: {status_text}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Hand + Face + Eye Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
