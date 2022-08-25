# Real-time detection of facial expressions

## About
Detect 6 different emotions from a human face in real-time (happy, sad, angry, surprised, scared, neutral), utilising the video feed from the users webcam. Results are displayed to the user via emoji's and text displayed in the corner of a video feed window.
<img src="https://github.com/systemvaz/Limbic/blob/master/FaceDetect/img/Demo.PNG" height="50%" width="50%" >
## Training dataset
The image dataset used for neural network training is FER2013, a collection of 35,887 images and related annotated emotions.
A train-test split of 90% and 10% respectively was used.
## Architecture
A ResNet164 convolutional neural network (CNN) is trained on the above dataset used for the detection.
OpenCV with frontal face Haar Cascade is utilised to detect and crop images of faces from the users video feed.
## Usage
* 1.) Download FER2013 data and place in folder Limbic/FaceDetect/dataset
* 2.) Run convert.py to generate *.png images from pixel data
* 3.) Run train.py to perform model training
* 4.) Run the main detection program detect.py
## Training results
![alt text](https://github.com/systemvaz/Limbic/blob/master/FaceDetect/img/training_results_resnet.png)
