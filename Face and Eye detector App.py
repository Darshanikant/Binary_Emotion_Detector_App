import streamlit as st
import cv2
import numpy as np


st.set_page_config(page_title="Face & Eye Detector", layout="centered")

# Title and description
st.title("Face & Eye Detector App 👁️‍🗨️")
st.write("This app uses Haar cascades to detect faces and eyes in real-time using your webcam.")

# Button to start detection
if st.button("Start Face & Eye Detection"):
    # Load Haar cascades
    face_classifier = cv2.CascadeClassifier(r"C:/Users/sunil/Downloads/haarcascade_frontalface_default.xml")
    eye_classifier = cv2.CascadeClassifier(r"C:/Users/sunil/Downloads/haarcascade_eye.xml")

    # Check if cascade files are loaded
    if face_classifier.empty() or eye_classifier.empty():
        st.error("Error: Haar cascade file not loaded. Check the file path.")
        st.stop()

    # Face detection function
    def face_detector(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        # If no faces detected, return the original image
        if len(faces) == 0:
            return img

        for (x, y, w, h) in faces:
            x = max(0, x - 50)  # Ensure x and y are not negative
            y = max(0, y - 50)
            w += 50
            h += 50
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Label Face
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_classifier.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
                cv2.putText(roi_color, "Eye", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Label Eye

        return img

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not access the webcam.")
        st.stop()

    # Display webcam feed in the Streamlit app
    FRAME_WINDOW = st.image([])

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to read frame from webcam.")
            break

        # Process frame and display it
        frame = cv2.cvtColor(face_detector(frame), cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        # Break loop if user stops the app
        if st.session_state.get("_is_running") == False:
            break

    cap.release()
    cv2.destroyAllWindows()
