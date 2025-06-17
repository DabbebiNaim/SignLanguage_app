# streamlit_app.py

import streamlit as st
from ultralytics import YOLO
import cv2
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- PAGE CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="ASL to Text Translator",
    page_icon="🤟", # You can use an emoji as an icon
    layout="wide"
)

# --- CONFIGURATION ---
# Update this path to your final, fine-tuned model
MODEL_PATH = 'runs/detect/asl_model_final_with_gestures2/weights/best.pt'
CONF_THRESHOLD = 0.7
DEBOUNCE_TIME = 1.0  # Seconds to wait before adding the same letter again

# --- MODEL LOADING ---
# Use Streamlit's caching to load the model only once
@st.cache_resource
def load_yolo_model(model_path):
    """Loads the YOLO model from the specified path."""
    return YOLO(model_path)

model = load_yolo_model(MODEL_PATH)


# --- SESSION STATE INITIALIZATION ---
# This ensures our variables persist across reruns
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'last_detected_char' not in st.session_state:
    st.session_state.last_detected_char = ""
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = 0
if 'run_webcam' not in st.session_state:
    st.session_state.run_webcam = False


# --- WEBRTC VIDEO TRANSFORMER ---
# This class processes each frame from the webcam
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # We don't need to re-initialize these here, they are in session_state
        pass

    def transform(self, frame):
        # Convert frame to a format OpenCV can use
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO inference
        results = model(img, conf=CONF_THRESHOLD)

        current_char = ""
        if results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                current_char = model.names[class_id]
                break  # Process only the most confident detection

        # --- Sentence Logic using Session State ---
        current_time = time.time()
        if current_char and current_char != "nothing":
            if current_char != st.session_state.last_detected_char or \
               (current_time - st.session_state.last_detection_time > DEBOUNCE_TIME):
                
                if current_char == 'space':
                    st.session_state.sentence += " "
                elif current_char == 'delete':
                    st.session_state.sentence = st.session_state.sentence[:-1]
                else:
                    st.session_state.sentence += current_char

                st.session_state.last_detected_char = current_char
                st.session_state.last_detection_time = current_time
        
        elif not current_char:
             st.session_state.last_detected_char = ""


        # Annotate the frame with detection boxes
        annotated_frame = results[0].plot()
        return annotated_frame



st.title("ASL to Text Translator")
st.markdown("This app uses a YOLOv8 model to translate American Sign Language gestures into text in real-time.")

# Create two columns: one for the webcam feed, one for controls and text
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Camera Feed")
    # Button to start the webcam
    if st.button("Start Webcam", key="start_cam"):
        st.session_state.run_webcam = True
    
    # The WebRTC streamer component
    if st.session_state.run_webcam:
        webrtc_streamer(
            key="example",
            video_transformer_factory=VideoTransformer,
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }),
            media_stream_constraints={"video": True, "audio": False},
        )
    else:
        st.info("Click 'Start Webcam' to begin translation.")

with col2:
    st.header("Translated Text & Controls")

    # Display the sentence in a text area
    st.text_area("Sentence:", st.session_state.sentence, height=200, key="sentence_display", disabled=True)

    # --- Control Buttons ---
    # The button logic updates the session state, and Streamlit reruns, updating the UI
    if st.button("Clear Sentence"):
        st.session_state.sentence = ""
        st.rerun()

    # The download button for saving the text
    st.download_button(
        label="Save Sentence",
        data=st.session_state.sentence.encode('utf-8'),
        file_name='asl_translation.txt',
        mime='text/plain',
    )
    
    # The exit button stops the webcam stream
    if st.button("Exit Interface"):
        st.session_state.run_webcam = False
        st.rerun()

st.sidebar.info(
    "Controls:\n"
    "- **Start Webcam**: Begins the video stream.\n"
    "- **Clear Sentence**: Erases the current text.\n"
    "- **Save Sentence**: Downloads the text as a .txt file.\n"
    "- **Exit Interface**: Stops the video stream."
)