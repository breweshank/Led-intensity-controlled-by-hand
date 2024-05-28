import cv2
import mediapipe as mp
import serial
import time
import math

# Initialize serial communication with Arduino
arduino = serial.Serial('COM15', 9600)  # Change 'COM3' to the appropriate port for your system
time.sleep(2)  # Wait for the serial connection to initialize

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

try:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame to avoid mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to find hands
        result = hands.process(rgb_frame)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get the coordinates of the thumb tip and index finger tip
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Convert coordinates to pixel values
                h, w, _ = frame.shape
                thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_tip_coords = (int(index_tip.x * w), int(index_tip.y * h))
                
                # Calculate the distance between the thumb tip and index finger tip
                distance = calculate_distance(thumb_tip_coords, index_tip_coords)
                
                # Map the distance to LED brightness (0-255)
                brightness = int(min(255, max(0, (distance / 200) * 255)))  # Adjust 200 based on max distance expected
                arduino.write(f'{brightness}\n'.encode())
                print(f"Distance: {distance}, Brightness: {brightness}")
                
                # Draw circles on thumb and index finger tips
                cv2.circle(frame, thumb_tip_coords, 10, (0, 255, 0), -1)
                cv2.circle(frame, index_tip_coords, 10, (0, 255, 0), -1)
                
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    # Close the serial connection
    arduino.close()
