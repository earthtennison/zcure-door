# from keras.engine import  Model
# from keras.layers import Input
# from keras_vggface.vggface import VGGFace
import cv2
import numpy as np
from PIL import Image
import time

# Convolution Features
#vgg_features = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling='avg')

#detect face
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    cropped_face = np.zeros(img.shape)
    (x0, y0, w0, h0) = (0,0,0,0)
    if len(faces)!=0:
        print(f"{len(faces)} face detected")
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
            (x0, y0, w0, h0) = (x, y, w, h)
    else:
        print("face not detected!")
    return cropped_face, (x0, y0, w0, h0)


cap = cv2.VideoCapture(0)
_,frame = cap.read()
input_size = (frame.shape[1],frame.shape[0])
while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, input_size)
        face, (x, y, h, w) = detect_face(frame)
        if((x,y,w,h)==(0,0,0,0)):
            continue
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imshow('video',frame)
        cv2.waitKey(1)    
    else:
        cap.release()
        break

