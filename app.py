import cv2
import tensorflow as tf
import numpy as np

# Load your trained TensorFlow model from the .keras file
model_path = 'weights/DenseNet2.h5'
model = tf.keras.models.load_model(model_path)

# Define video file path
video_path = 'video/videoplayback.mp4'
# video_path = 0 # for webcam

# Define a function for preprocessing frames
def preprocess_frame(frame):
    # Resize frame to match model input size
    resized_frame = cv2.resize(frame, (224, 224))
    # Normalize pixel values
    normalized_frame = resized_frame / 255.0  # Assuming pixel range [0, 255]
    return normalized_frame

# Open the video file
video = cv2.VideoCapture(video_path)

# Loop through each frame in the video
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    # Preprocess the frame
    input_frame = preprocess_frame(frame)
    
    # Perform inference
    input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension
    probabilities = model.predict(input_frame)
    predicted_class = np.argmax(probabilities)
    
    prob = np.round(probabilities[0][predicted_class] * 100, 2)
    
    
    # Display the frame with prediction
    if prob >= 90:  # Accident detected
        cv2.putText(frame, 'Accident Detected: ' + str(prob) , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, 'No Accident: ' + str(prob), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Video', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close windows
video.release()
cv2.destroyAllWindows()