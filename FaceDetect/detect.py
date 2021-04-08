# ----------------------------------------------
# Author: Alex Varano
# Main video capture and expression detection code
# utilising model saved by training.py
# ----------------------------------------------

from cv2 import cv2
import sys
import os
import numpy as np
import tensorflow as tf

def get_model_paths():
    cascPath = os.curdir + '/saved_models/haarcascade_frontalface_default.xml'
    modelPath = os.curdir + '/saved_models/EmoDetect_ResNet164v2_model.068.h5'
    
    return cascPath, modelPath


def get_emoji_paths():
    img_path = os.curdir + '/img/emojis/'
    angry_emj = cv2.imread(img_path + 'angry.png')
    disgust_emj = cv2.imread(img_path + 'disgust.png')
    fear_emj = cv2.imread(img_path + 'fear.png')
    happy_emj = cv2.imread(img_path + 'happy.png')
    sad_emj = cv2.imread(img_path + 'sad.png')
    surprised_emj = cv2.imread(img_path + 'surprised.png')
    neutral_emj = cv2.imread(img_path + 'neutral.png')
    
    return angry_emj, disgust_emj, fear_emj,\
           happy_emj, sad_emj, surprised_emj, neutral_emj


def get_dicts(angry_emj, disgust_emj, fear_emj, happy_emj, sad_emj, surprised_emj, neutral_emj):
    classes_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 
                    4:'Sad', 5:'Surprised', 6:'Neutral'}
    emoji_dict   = {0:angry_emj, 1:disgust_emj, 2:fear_emj, 3:happy_emj, 
                    4:sad_emj, 5:surprised_emj, 6:neutral_emj}

    return classes_dict, emoji_dict


def capture_and_detect(video_capture, faceCascade, model, emoji_dict, classes_dict):
    while True:
        # Capture frame-by-frame
        _, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(48, 48),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop face and predict expression
        try:
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            crop_img = crop_img.astype('float32') / 255
            crop_img = cv2.resize(crop_img, (48, 48))
            crop_img = crop_img.reshape(1,48,48,1)
            expression = model.predict(crop_img)
            expression = int(np.argmax(expression, axis = 1))
            overlay = emoji_dict.get(expression, neutral_emj)
            result = classes_dict.get(expression, "Don't know")
        except:
            print("Problem getting facial expression...")

        # Add emoji and text to top left corner
        rows, cols, _ = overlay.shape
        frame[0:rows, 0:cols ] = overlay
        cv2.putText(frame, result, (rows+30, cols//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (209, 80, 0, 255), 3)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    # Get filepath declarations
    cascPath, modelPath = get_model_paths()

    angry_emj, disgust_emj, fear_emj,\
    happy_emj, sad_emj, surprised_emj, neutral_emj = get_emoji_paths()

    classes_dict, emoji_dict = get_dicts(angry_emj, disgust_emj, fear_emj,
                                         happy_emj, sad_emj, surprised_emj, neutral_emj)

    # Load model and face detection algo
    model = tf.keras.models.load_model(modelPath, compile=True)
    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture(0)

    # Main video capture loop
    capture_and_detect(video_capture, faceCascade, model, emoji_dict, classes_dict)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()