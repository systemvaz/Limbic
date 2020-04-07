import cv2
import sys
import os
import numpy as np
import tensorflow as tf

cascPath = os.curdir + '/models/haarcascade_frontalface_default.xml'
modelPath = os.curdir + '/models/EmoDetect_ResNet164v2_model.068.h5'

angry_emj = cv2.imread(os.curdir + '/emojis/angry.png')
disgust_emj = cv2.imread(os.curdir + '/emojis/disgust.png')
fear_emj = cv2.imread(os.curdir + '/emojis/fear.png')
happy_emj = cv2.imread(os.curdir + '/emojis/happy.png')
sad_emj = cv2.imread(os.curdir + '/emojis/sad.png')
surprised_emj = cv2.imread(os.curdir + '/emojis/surprised.png')
neutral_emj = cv2.imread(os.curdir + '/emojis/neutral.png')

classes_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 
                4:'Sad', 5:'Surprised', 6:'Neutral'}

emoji_dict = {0:angry_emj, 1:disgust_emj, 2:fear_emj, 3:happy_emj, 
              4:sad_emj, 5:surprised_emj, 6:neutral_emj}

model = tf.keras.models.load_model(modelPath, compile=True)
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

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
    rows, cols, channels = overlay.shape
    frame[0:rows, 0:cols ] = overlay
    cv2.putText(frame, result, (rows+30, cols//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (209, 80, 0, 255), 3)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()