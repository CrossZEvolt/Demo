import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể kết nối với webcam.")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Không thể lấy khung hình.")
        break

    # Chuyển đổi khung hình từ BGR sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Thực hiện phát hiện bàn tay
    results = hands.process(frame_rgb)

    # Kiểm tra xem có bàn tay nào được phát hiện hay không
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ các điểm mốc trên bàn tay
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Hiển thị khung hình đã xử lý
    cv2.imshow('Hand Gesture Recognition', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và đóng kết nối
cap.release()
cv2.destroyAllWindows()
def is_hand_open(hand_landmarks):
    # Xác định các điểm mốc cho ngón cái, ngón trỏ, ngón giữa, ngón áp út và ngón út
    finger_tips = [8, 12, 16, 20]  # Các chỉ số cho các đầu ngón tay (trừ ngón cái)

    # Kiểm tra các đầu ngón tay có ở phía trên các đốt ngón tay tương ứng không
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip - 2].y:
            return False
    return True
def is_hand_closed(hand_landmarks):
    # Xác định các điểm mốc cho ngón cái, ngón trỏ, ngón giữa, ngón áp út và ngón út
    finger_tips = [8, 12, 16, 20]
# Kiểm tra xem các đầu ngón tay có ở phía dưới các đốt ngón tay tương ứng không
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            return False
    return True
def is_v_sign(hand_landmarks):
    # Kiểm tra ngón trỏ và ngón giữa đang duỗi thẳng
    index_finger = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_finger = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y

    # Kiểm tra các ngón còn lại đang nắm lại
    other_fingers = (hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and
                     hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y)

    return index_finger and middle_finger and other_fingers
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Kiểm tra xem có bàn tay nào được phát hiện không
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Kiểm tra và hiển thị cử chỉ
            if is_hand_open(hand_landmarks):
                gesture_text = "Open Hand"
            elif is_hand_closed(hand_landmarks):
                gesture_text = "Closed Hand"
            elif is_v_sign(hand_landmarks):
                gesture_text = "V Sign"
            else:
                gesture_text = "Unknown Gesture"

            # Hiển thị tên cử chỉ lên khung hình
            cv2.putText(frame, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Hiển thị khung hình
    cv2.imshow('Hand Gesture Recognition', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

