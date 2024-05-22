# import base64
# import os  # Import the os module

# import cv2
# import numpy as np
# import streamlit as st
# from PIL import Image

# # Load pre-trained face detection model (Haar cascades)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# class SkinDetector:
#     def detect_skin(self, image):
#         # Convert image to HSV color space
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         # Define lower and upper bounds for skin color in HSV
#         lower_skin = np.array([0, 48, 80], dtype=np.uint8)
#         upper_skin = np.array([20, 255, 255], dtype=np.uint8)
#         # Threshold the HSV image to get only skin color
#         mask = cv2.inRange(hsv, lower_skin, upper_skin)
#         # Apply morphological operations to remove noise
#         kernel = np.ones((5, 5), np.uint8)
#         mask = cv2.erode(mask, kernel, iterations=1)
#         mask = cv2.dilate(mask, kernel, iterations=2)
#         return mask

# class DeepfakeDetector:
#     def _init_(self):
#         self.skin_detector = SkinDetector()

#     def detect_deepfake(self, file_data, file_type):
#         if file_data is not None:
#             if file_type == 'image':
#                 return self.process_image(file_data)
#             elif file_type == 'video':
#                 return self.process_video(file_data)
#             elif file_type == 'audio':
#                 return self.process_audio(file_data)

#     def process_image(self, file_data):
#         img = np.array(bytearray(file_data.read()), dtype=np.uint8)
#         img = cv2.imdecode(img, cv2.IMREAD_COLOR)
#         # Detect faces using Haar cascades
#         faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#         # Draw rectangles around the faces
#         for (x, y, w, h) in faces:
#             cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
#         # Detect skin
#         skin_mask = self.skin_detector.detect_skin(img)
#         img_with_skin = cv2.bitwise_and(img, img, mask=skin_mask)
#         return img_with_skin, len(faces) > 0

#     def process_video(self, file_data):
#         is_real = True
#         # Convert the file_data into bytes
#         video_bytes = file_data.read()

#         # Save the video bytes to a temporary file
#         with open("temp_video.mp4", "wb") as f:
#             f.write(video_bytes)

#         # OpenCV requires a file path, so we use the temporary file path
#         cap = cv2.VideoCapture("temp_video.mp4")

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Detect faces
#             faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#             # Draw rectangles around the faces
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

#             # Detect skin
#             skin_mask = self.skin_detector.detect_skin(frame)
#             frame_with_skin = cv2.bitwise_and(frame, frame, mask=skin_mask)

#             # Display the frame
#             cv2.imshow('Video', frame_with_skin)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#             if len(faces) > 0:
#                 is_real = False

#         # Release resources and delete the temporary file
#         cap.release()
#         cv2.destroyAllWindows()
#         os.remove("temp_video.mp4")  # Remove the temporary video file

#         return None, is_real

#     def process_audio(self, file_data):
#         # Placeholder for audio processing
#         return False

# def main():
#     detector = DeepfakeDetector()

#     st.title("Deepfake Detection")
#     st.sidebar.title("INPUT ")
#     detection_mode = st.sidebar.selectbox("Select Detection Mode", ("Image", "Video", "Audio"))

#     if detection_mode == "Image":
#         st.subheader("Image Deepfake Detection")
#         image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
#         if image_file is not None:
#             if st.button("Detect"):
#                 result, is_real = detector.detect_deepfake(image_file, 'image')
#                 if is_real:
#                     st.image(result, caption="Real Image", channels="BGR", use_column_width=True)
#                 else:
#                     st.image(result, caption="Fake Image", channels="BGR", use_column_width=True)

#     elif detection_mode == "Video":
#         st.subheader("Video Deepfake Detection")
#         video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
#         if video_file is not None:
#             if st.button("Detect"):
#                 _, is_real = detector.detect_deepfake(video_file, 'video')
#                 if is_real:
#                     st.write("The video is real.")
#                 else:
#                     st.write("The video is fake.")

#     # elif detection_mode == "Audio":
#     #     st.subheader("Audio Deepfake Detection")
#     #     st.write("Audio detection feature is not implemented yet.")

# if __name__ == "_main_":
#     main()

# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()
# # ---------------------BACKGORUND IMAGE-------------------------
# bg_image = Image.open("image3.png")

# page_bg_img = '''
# <style>
# .stApp {
#   background-image: url("data:image3/png;base64,%s");
#   background-size: cover;
# }
# </style>
# ''' % get_base64_of_bin_file("image3.png")


# with open("image1.png", "rb") as f:
#     data = base64.b64encode(f.read()).decode("utf-8")

# st.sidebar.header("PERFORMANCE METRICS ACROSS FOLDS")
# st.sidebar.markdown("Metric value VS Fold ")

# st.sidebar.markdown(
#     f"""
#     <div style="display:table;margin-top:5%;margin-left:5%,margin-right:5%;">
#         <img src="data:image1/png;base64, {data}" width="300" height="200">
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# with open("image2.png", "rb") as f:
#     data = base64.b64encode(f.read()).decode("utf-8")

# st.sidebar.header("EPOCH VS ACCURACY")
# # st.sidebar.markdown("Metric value VS Fold ")

# st.sidebar.markdown(
#     f"""
#     <div style="display:table;margin-top:5%;margin-left:5%,margin-right:5%;">
#         <img src="data:image2/png;base64, {data}" width="300" height="200">
#     </div>
#     """,
#     unsafe_allow_html=True,
# )