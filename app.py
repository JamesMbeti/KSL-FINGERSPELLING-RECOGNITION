# Importing necessary libraries
import cv2
import time
import os
import uuid
import math
import base64
import skimage
import tempfile
import tensorflow
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import mediapipe as mp
from streamlit_lottie import st_lottie
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import array_to_img, img_to_array
from cvzone.HandTrackingModule import HandDetector as hd
from cvzone.ClassificationModule import Classifier


#requirements
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.hands
best_model = "best_model.h5"

# Set page configuration
st.set_page_config(page_title="The ODA Clan", page_icon="👽", layout="wide",
                   initial_sidebar_state="expanded")

#load assets
cover_image = Image.open("fingerspell.jpeg")
DEMO_IMAGE = 'WhatsApp Image 2023-07-08 at 11.08.15.jpeg'
DEMO_VIDEO = "WhatsApp Video 2023-07-11 at 12.40.11.mp4"

#loading the model
model = load_model("best_model.h5")

word_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F",
             6: "G", 7: "H", 8: "I", 9: "K", 10: "L", 11: "M",
             12: "N", 13: "O", 14: "P", 15: "Q", 16: "R", 17: "S",
             18: "T", 19: "U", 20: "V", 21: "W", 22: "X", 23: "Y"}


#Global variable to store the video capture boject
cap = None

def video_capture():

    """A function that captures images and predicts realtime and saves images for word prediction"""

    global cap
    #Index of the webcam
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

    #getting the detector
    detector = hd(maxHands=1)
    classifier = Classifier("keras_model.h5", "labels.txt")

    #helps to ensure there's enough space for the gestures
    offset = 20
    imgSize = 600
    counter = 0

    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", 
              "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
    img_arrays = []
    while True:
        success, img = cap.read()
        if not success:
            continue
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h/w

            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[0:, wGap:wCal + wGap] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
    
    
            cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x - offset + 90, y - offset),
                        (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y-26), cv2.FONT_HERSHEY_COMPLEX, 2,
                        (255,255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                        (x + w +offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("Image", imgOutput)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            counter += 1
            img_arrays.append(labels[index])
    cap.release()
    cv2.destroyAllWindows()
    return img_arrays

def predict_web_word(img_array):
    return "".join(img_array).capitalize()


def preprocess_images(image):
    """A function that outputs a preprocessed image"""
    img_array = img_to_array(image)
    img_array /= 255
    img_array = cv2.resize(img_array, (64,64))
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    gray = np.expand_dims(gray, 2)
    image = skimage.filters.gaussian(gray, sigma=1)
    image = skimage.exposure.equalize_hist(image)
    image = image.reshape(-1, 64, 64, 1)
    return image


def extract_frames_from_video(video_path, start_time, end_time, interval):

    """A function that extracts pictures from a video"""

    img_arrays = []
     # Load video file
    cap = cv2.VideoCapture(video_path)
     # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), int(total_frames))
     # Set frame interval
    frame_interval = int(interval * fps)
     # Set starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
     # Initialize variables
    frame_count = 0
    current_frame = start_frame
    while current_frame <= end_frame:
         # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
         # Save frame as an image

        if frame_interval != 0 and frame_count % frame_interval == 0:
             # image_path = f"{output_folder}/frame_{current_frame}.jpg"
             # cv2.imwrite(image_path, frame)
            array = img_to_array(frame)
            img_arrays.append(array/255)
         # Update variables
        frame_count += 1
        current_frame += 1
     # Release the video capture
    cap.release()
    return img_arrays


def preprocess_video_images(img_array):

    """A function that takes in a list of image arrays and do preprocessing to the images"""

    images_array = []
    for img_file in img_array:
        height, width = img_file.shape[:2]
        new_width = int(width * 10)
        new_height = int(height * 10)
        zoomed_image = cv2.resize(img_file, (new_width, new_height))
        img_array = cv2.resize(zoomed_image, (64,64))
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, 2)
        image = skimage.filters.gaussian(gray, sigma=1)
        image = skimage.exposure.equalize_hist(image)
        images_array.append(image)

    word_images = np.array(images_array)
    return word_images

def predict_word(img_array):

    """ A function that takes in  an array of images and outputs a word"""

    word_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F",
             6: "G", 7: "H", 8: "I", 9: "K", 10: "L", 11: "M",
             12: "N", 13: "O", 14: "P", 15: "Q", 16: "R", 17: "S",
             18: "T", 19: "U", 20: "V", 21: "W", 22: "X", 23: "Y"}
    word_preds = model.predict(img_array)
    pred_series = pd.Series([np.argmax(x) for x in word_preds])
    predicted_word_series = pred_series.map(word_dict)
    predicted_word = "".join(predicted_word_series.values)
    return predicted_word.capitalize()


st.title('KSL HAND GESTURE RECOGNITION')

st.markdown(
    """
<style>
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
    width: 350px
}
[data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
    width: 350px;
    margin-left: -350px;
}
</style>
""",
unsafe_allow_html=True)

st.sidebar.title('KSL HAND RECOGNITION SIDEBAR')
st.sidebar.subheader('Parameters')

@st.cache
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    din = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = width / float(w)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

app_mode = st.sidebar.selectbox("Choose the app mode",
                                ["About App", "Run on image", "Run on video",
                                 "Run on WebCam", "Feedback"])

if app_mode == 'About App':
    st.image(cover_image, width=750)
    st.markdown("""
                This application offers an interface that allows the conversion of
                 Kenya Sign Language **KSL** fingerspells to their appropriate alphabets.
                """)
    st.markdown("---")
    st.markdown("")

elif app_mode == 'Feedback':
    # Contact
    st.write("---")
    st.header("We would love to hear from you!")

    contact_form = """
    <style>
    .contact-form {
        max-width: 400px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f5f5f5;
        border-radius: 5px;
    }
    .contact-form label {
        font-weight: bold;
        color: #333;
    }
    .contact-form input,
    .contact-form textarea,
    .contact-form button {
        margin-bottom: 10px;
        width: 100%;
        padding: 10px;
        border-radius: 3px;
        border: 1px solid #ccc;
    }
    .contact-form textarea {
        resize: vertical;
        min-height: 100px;
    }
    .contact-form button {
        padding: 10px;
        background-color: #4CAF50;
        color: white;
        border: none;
        cursor: pointer;
    }
    
    /* Styles for printing in black and white mode */
    @media print {
        .contact-form label {
            color: black !important;
        }
        .contact-form input,
        .contact-form textarea,
        .contact-form button {
            background-color: white !important;
            color: black !important;
            border: 1px solid black !important;
        }
    }
    </style>

    <div class="contact-form">
        <form action="https://formsubmit.co/briankitainge@gmail.com" method="POST">
            <label for="name">Name</label>
            <input type="text" name="name" id="name" required>
            <label for="email">Email</label>
            <input type="email" name="email" id="email" required>
            <label for="message">Message</label>
            <textarea name="message" id="message" required></textarea>
            <button type="submit" key="feedback_submit">Send</button>
        </form>
    </div>
    """

    st.markdown(contact_form, unsafe_allow_html=True)



elif app_mode == 'Run on image':
    st.title("Image Prediction")
    img_file_buffer = st.file_uploader('Upload an Image', type=["jpg", "jpeg", "png"])
    if img_file_buffer is not None:
        image_uploaded = Image.open(img_file_buffer)
        max_width = 200 
        image_uploaded.thumbnail((max_width, max_width), Image.ANTIALIAS)
        st.image(image_uploaded, caption="Uploaded Image")
        preprocessed_image = preprocess_images(image_uploaded)
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction)

        st.write("Predicted Letter: ", word_dict[predicted_class])

    else:
        st.write("No image file uploaded.")
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))
        st.sidebar.text("Preview")
        st.sidebar.image(image)

elif app_mode == 'Run on video':
    st.title("video Prediction")
    st.write("Upload a video for prediction")
    interval_time = st.number_input("The interval Between signs", min_value=0.5)
    start = st.number_input("The video starts")
    end = st.number_input("The video ends")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
  
    if uploaded_video is not None:
        video_uploaded = uploaded_video.read()
        
        #Displaying the video
        video_html = f"""
        <video width="100%" height="300" controls><source src="data:video/mp4;base64,
        {base64.b64encode(video_uploaded).decode()}" type="video/mp4"></video>
        """
        st.markdown(video_html, unsafe_allow_html=True)
        video_path = "./uploaded_video.mp4"
        with open(video_path, "wb") as video_writer:
            video_writer.write(video_uploaded)
        video_image_arrays = extract_frames_from_video(video_path=video_path, start_time=start,
                                                       end_time=end, interval=interval_time)
        processed_images = preprocess_video_images(video_image_arrays)
        if st.button("Predict"):
            predicted_word = predict_word(processed_images)
            st.write("Predicted Word: ", predicted_word)

    else:
        st.write("No video file uploaded.")

elif app_mode == "Run on WebCam":
    st.title("Web Prediction")
    st.write("""To save an image for prediction press "s" and to exit from the webcam press "q" """)
    
    if st.button("Start Webcam"):
        word_list = video_capture()
        word = predict_web_word(word_list)
        st.write("Predicted Word: ", word)
        if st.button("Stop Capture"):
            cap.release()
            cv2.destroyAllWindows()
    
    else:
        st.write("press above to start the camera")


