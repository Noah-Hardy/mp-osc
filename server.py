import time
import json
import cv2
import mediapipe as mp
from pythonosc import udp_client

# Function to get the bounds of pose landmarks with their values
# This function calculates the maximum and minimum x, y, z coordinates of the pose landmarks
# and returns a dictionary with the details of the landmarks at those bounds.
def get_pose_bounds_with_values(results):
    max_x = max_y = max_z = float('-inf')
    min_x = min_y = min_z = float('inf')
    max_x_idx = max_y_idx = min_x_idx = min_y_idx = max_z_idx = min_z_idx = -1

    # Iterate through all pose landmarks to find bounds
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
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
        if landmark.z > max_z:
            max_z = landmark.z
            max_z_idx = idx
        if landmark.z < min_z:
            min_z = landmark.z
            min_z_idx = idx

    # Return a dictionary with the bounds and landmark details
    return {
        "max_x": landmark_dict(results, max_x_idx),
        "min_x": landmark_dict(results, min_x_idx),
        "max_y": landmark_dict(results, max_y_idx),
        "min_y": landmark_dict(results, min_y_idx),
        "max_z": landmark_dict(results, max_z_idx),
        "min_z": landmark_dict(results, min_z_idx)
    }

# Function to convert a landmark to a dictionary with its details
# This function extracts the x, y, z coordinates and visibility of a landmark
# and returns a dictionary with its id and values.
def landmark_dict(results,idx):
    lm = results.pose_landmarks.landmark[idx]
    return {
        "id": idx,
        "x": lm.x,
        "y": lm.y,
        "z": lm.z,
        "visibility": getattr(lm, "visibility", None)
    }

# Create an OSC client to send messages to a specified IP and port
# Change the IP and port as needed for your OSC server
osc_client = udp_client.SimpleUDPClient("192.168.1.28", 1234)

# Initialize Mediapipe drawing utilities and holistic model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Open a video capture stream
# Change the camera index if needed (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Start the Mediapipe holistic model for pose and hand detection
# Adjust the detection and tracking confidence as needed
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

        # Convert the image back to BGR for regular projection
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Gather all landmarks
        pose_landmarks = []
        right_hand_landmarks = []
        left_hand_landmarks = []
        timestamp = time.time()

        # Pose landmarks
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                pose_landmarks.append({
                    "type": "pose",
                    "id": idx,
                    "x": round(landmark.x, 2),
                    "y": round(landmark.y, 2),
                    "z": round(landmark.z, 2),
                    "visibility": round(landmark.visibility, 2) if hasattr(landmark, "visibility") else None
                })
                # Draw the landmark ID on the image
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(frame, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            

        # Right hand landmarks
        if results.right_hand_landmarks:
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                right_hand_landmarks.append({
                    "type": "right_hand",
                    "id": idx,
                    "x": round(landmark.x, 2),
                    "y": round(landmark.y, 2),
                    "z": round(landmark.z, 2)
                })
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(frame, f"RH-{idx}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        # Left hand landmarks
        if results.left_hand_landmarks:
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                left_hand_landmarks.append({
                    "type": "left_hand",
                    "id": idx,
                    "x": round(landmark.x, 2),
                    "y": round(landmark.y, 2),
                    "z": round(landmark.z, 2)
                })
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(frame, f"LH-{idx}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        # Send all landmarks on a single OSC channel
        if pose_landmarks:
            osc_payload = {
                "timestamp": timestamp,
                "landmarks": pose_landmarks
            }
            bounds = get_pose_bounds_with_values(results)
            osc_client.send_message("/pose", json.dumps(osc_payload))
            osc_client.send_message("/bounds", json.dumps(bounds))

        if right_hand_landmarks:
            osc_payload = {
                "timestamp": timestamp,
                "landmarks": right_hand_landmarks
            }
            osc_client.send_message("/right_hand", json.dumps(osc_payload))

        if left_hand_landmarks:
            osc_payload = {
                "timestamp": timestamp,
                "landmarks": left_hand_landmarks
            }
            osc_client.send_message("/left_hand", json.dumps(osc_payload))

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