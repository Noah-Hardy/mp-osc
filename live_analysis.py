import mediapipe as mp
import cv2
import time
from pythonosc import udp_client
import json

def get_pose_bounds_with_values(landmarks):
    max_x = max_y = float('-inf')
    min_x = min_y = float('inf')
    max_x_idx = max_y_idx = min_x_idx = min_y_idx = -1

    for idx, landmark in enumerate(landmarks):
        if landmark.x > max_x:
            max_x = landmark.x
            max_x_idx = idx
        if landmark.x < min_x:
            min_x = landmark.x
            min_x_idx = idx
        if landmark.y > max_y:
            max_y = landmark.y
            max_y_idx = idx
        if landmark.y < min_y:
            min_y = landmark.y
            min_y_idx = idx

    def landmark_dict(idx):
        lm = landmarks[idx]
        return {
            "id": idx,
            "x": lm.x,
            "y": lm.y,
            "z": lm.z,
            "visibility": getattr(lm, "visibility", None)
        }

    return {
        "max_x": landmark_dict(max_x_idx),
        "min_x": landmark_dict(min_x_idx),
        "max_y": landmark_dict(max_y_idx),
        "min_y": landmark_dict(min_y_idx)
    }



osc_client = udp_client.SimpleUDPClient("192.168.1.28", 1234)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_objectron = mp.solutions.objectron

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB for Mediapipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image using Mediapipe
        results = holistic.process(image)
        # objectron_results = objectron.process(image)
        
        # Convert the image back to BGR for regular projection
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        	
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract and send pose landmarks
        if results.pose_landmarks:
            timestamp = time.time()
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_data = {
                    "timestamp": timestamp,
                    "id": idx,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                }
                # Send each landmark on a separate OSC channel identified by idx
                osc_client.send_message(f"/pose/{idx}", json.dumps(landmark_data))

                # Draw the landmark ID on the image
                h, w, _ = frame.shape  # Get image dimensions
                cx, cy = int(landmark.x * w), int(landmark.y * h)  # Convert normalized coordinates to pixel values
                cv2.putText(frame, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        # Calculate and send pose bounds with values
            bounds = get_pose_bounds_with_values(pose_landmarks)
            osc_client.send_message("/pose/bounds", json.dumps(bounds))

        if results.face_landmarks:
            face_data = []
            for idx, landmark in enumerate(results.face_landmarks.landmark):
                face_data.append({
                    "id": idx,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                })

         # Extract and send right hand landmarks
        if results.right_hand_landmarks:
            timestamp = time.time()
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                landmark_data = {
                    "timestamp": timestamp,
                    "id": idx,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                }
                # Send each landmark on a separate OSC channel identified by idx
                osc_client.send_message(f"/right_hand/{idx}", json.dumps(landmark_data))

                # Draw the landmark ID on the image
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(frame, f"RH-{idx}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        # Extract and send left hand landmarks
        if results.left_hand_landmarks:
            timestamp = time.time()
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                landmark_data = {
                    "timestamp": timestamp,
                    "id": idx,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                }
                # Send each landmark on a separate OSC channel identified by idx
                osc_client.send_message(f"/left_hand/{idx}", json.dumps(landmark_data))

                # Draw the landmark ID on the image
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(frame, f"LH-{idx}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        # Display the image with pose and hand landmarks drawn
        mp.solutions.drawing_utils.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        mp.solutions.drawing_utils.draw_landmarks(
            frame, 
            results.right_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        mp.solutions.drawing_utils.draw_landmarks(
            frame, 
            results.left_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )
        

        cv2.imshow('Pose and Hand Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()