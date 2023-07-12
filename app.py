# import cv2
# import streamlit as st
# import numpy as np
# from tensorflow.keras.models import load_model
# import skimage
# import time

# model = load_model("best_model.h5")
# interval = 2
# image_placeholder = st.empty()

# captured_images = []

# class_to_letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F",
#                    6: "G", 7: "H", 8: "I", 9: "K", 10: "L", 11: "M",
#                    12: "N", 13: "O", 14: "P", 15: "Q", 16: "R", 17: "S",
#                    18: "T", 19: "U", 20: "V", 21: "W", 22: "X", 23: "Y"}
# def capture_image():
#     camera = cv2.VideoCapture(0)
#     if not camera.isOpened():
#         st.error("Failed to open the camera")
#         return None
#     ret, frame = camera.read()
#     if not ret:
#         st.error("Failed to capture frame from the camera")
#         return None
#     camera.release()
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     return frame_rgb
    

# def preprocess_image(image):
#     image = cv2.resize(image, (64,64))
#     image = image.astype(np.float32) / 255.0
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = np.expand_dims(gray, 2)
#     image = skimage.filters.gaussian(gray, sigma=1)
#     image = skimage.exposure.equalize_hist(image)
#     return image

# def predict_image(image):
#     preprocessed_image = preprocess_image(image)
#     prediction = model.predict(preprocessed_image)
#     predicted_class = np.argmax(prediction)
#     return predicted_class

# def main():
#     st.title("Camera Capture and Prediction")
#     if st.button("Start Capture"):
#         st.write("capturing images...")
#         while st.button("Stop Capture", key="stop_capture"):
#             st.write("Capturing image...")
#             captured_image = capture_image()

#             if captured_image is not None:
#                 image_placeholder.image(captured_image, channels="RGB", use_column_width=True)

#                 predicted_class = predict_image(captured_image)

#                 if predicted_class in class_to_letter:
#                     predicted_letter = class_to_letter[predicted_class]
#                 else:
#                     predicted_letter = 'Unknown'

#                 captured_images.append(predicted_letter)

#             # Wait for the specified interval
#             time.sleep(interval)

#     # Display the captured letters as a single word
#     if captured_images:
#         word = ''.join(captured_images)
#         st.write("Captured Word:", word)


# if __name__ == '__main__':
#     main()


import cv2
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import skimage
import time

model = load_model("best_model.h5")
interval = 2
image_placeholder = st.empty()

captured_images = []

class_to_letter = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F",
    6: "G", 7: "H", 8: "I", 9: "K", 10: "L", 11: "M",
    12: "N", 13: "O", 14: "P", 15: "Q", 16: "R", 17: "S",
    18: "T", 19: "U", 20: "V", 21: "W", 22: "X", 23: "Y"
}

def capture_image():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        st.error("Failed to open the camera")
        return None
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to capture frame from the camera")
        return None
    camera.release()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

def preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    image = image.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.expand_dims(gray, 2)
    image = skimage.filters.gaussian(gray, sigma=1)
    image = skimage.exposure.equalize_hist(image)
    return image

def predict_image(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    predicted_class = np.argmax(prediction)
    return predicted_class

def main():
    st.title("Camera Capture and Prediction")
    if st.button("Start Capture", key="Start"):
        st.write("Capturing images...")
        # while not st.button("Stop Capture", key="Stop_capture"):
        #     st.write("Capturing image...")
        captured_image = capture_image()

        if captured_image is not None:
            image_placeholder.image(captured_image, channels="RGB", use_column_width=True)

            predicted_class = predict_image(captured_image)

            if predicted_class in class_to_letter:
                predicted_letter = class_to_letter[predicted_class]
            else:
                predicted_letter = 'Unknown'

            captured_images.append(predicted_letter)

            # Wait for the specified interval
        time.sleep(interval)

    # Display the captured letters as a single word
    if captured_images:
        word = ''.join(captured_images)
        st.write("Captured Word:", word)

if __name__ == '__main__':
    main()
