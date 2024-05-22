import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape

        # Make detection
        results = pose.process(image)

        # Render detections
        if results.pose_landmarks:
            # Get landmarks
            landmarks = results.pose_landmarks.landmark

            # Get left and right shoulder landmarks
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image_width, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * image_height]

            # Calculate the difference between left and right shoulder positions
            shoulder_difference = abs(left_shoulder[1] - right_shoulder[1]) / image_height * 100

            # Check if the shoulder line is above or below 60% of the screen
            if left_shoulder[1] < 0.6 * image_height or right_shoulder[1] < 0.6 * image_height:
                # Incorrect posture, show alert
                cv2.putText(image, "Incorrect Posture", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif shoulder_difference > 5:
                # Shoulder height difference is more than 5%, show alert
                cv2.putText(image, "Shoulder Height Difference!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Correct posture, show in green
                cv2.putText(image, "Correct Posture", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Seating Posture Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()