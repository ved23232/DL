import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import speech_recognition as sr
import threading

# Class mapping for YOLO model
class_mapping = {
    0: 'ThumbsUp',
    1: 'ThumbsDown',
    2: 'ThankYou',
    3: 'LiveLong',
    4: 'Mother',
    5: 'Father',
    6: 'I',
    7: 'Yes',
    8: 'No',
    9: 'Help',
    10: 'Food',
    11: 'More',
    12: 'Bathroom',
    13: 'Fine',
    14: 'Repeat'
}

# Initialize YOLO model
model = YOLO('DL/best_200.pt')

# Function to perform sign language recognition
def recognize_sign_language_and_speech():
    st.markdown('<h1 style="text-align: center;">Sign Language Recognition</h1>', unsafe_allow_html=True)

    cap = cv2.VideoCapture(0)
    recognizer = sr.Recognizer()

    while cap.isOpened() and not quit_button:
        ret, frame = cap.read()
        if not ret:
            break

        if speech_recognition_enabled:
            with sr.Microphone() as source:
                st.button("Start Speaking")

                audio = recognizer.listen(source)
                st.write("Recognizing...")

                try:
                    text = recognizer.recognize_google(audio)
                    st.write(f"You said: {text}")
                except sr.UnknownValueError:
                    st.write("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    st.write(f"Could not request results from Google Speech Recognition; {e}")

        # Use YOLO model to make predictions on the frame
        results = model.predict(frame)

        bbox = results[0].boxes.xyxy  # Extract bounding boxes

        for box in bbox:
            x_min, y_min, x_max, y_max = box[0].item(), box[1].item(), box[2].item(), box[3].item()

            values = results[0].boxes.cls
            confidences = results[0].boxes.conf

            labels = []
            confidence = []

            for i in range(len(values)):
                value = values[i]
                conf = confidences[i]
                labels.append(value)
                confidence.append(conf)

            if labels:
                max_confidence_idx = np.argmax(confidence)
                max_confidence_label = labels[max_confidence_idx]
                max_confidence = confidence[max_confidence_idx]
            else:
                max_confidence_label = 'Unknown'
                max_confidence = 0.0

            # Display the frame with labels and recognized speech
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            label_text = f'{class_mapping.get(int(max_confidence_label))} ({max_confidence:.2f})'
            cv2.putText(frame, label_text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the combined frame using st.image
        video_placeholder.image(frame, channels="BGR")

    if quit_button:
        cap.release()
        cv2.destroyAllWindows()
        st.stop()

# Initialize Streamlit app
st.title("Bidirectional Multimodal Sign Recognition System")

video_placeholder = st.empty()

quit_button = st.button('Quit')
sign_button = st.button("Start Sign Language Recognition")
speech_recognition_enabled = False

# Add a button to start sign language recognition
if sign_button:
    recognize_sign_language_and_speech()
