import cv2
import numpy as np
from PIL import Image
import time
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

def preprocess(img_path, is_cropped = False):
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if not is_cropped:
        face ,_ = detect_face(img)
    else:
        face = img
    face = cv2.resize(face,(224,224))
    face=np.array((face,))
    return face

def preprocess_video(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face, (x, y, w, h) = detect_face(img)
    face = cv2.resize(face,(224,224))
    face=np.array((face,))
    return face, (x, y, w, h)

im_path = "sample_image/dene_regis.jpg"
img = cv2.imread(im_path)
cv2.imshow('image_original',img)
cv2.waitKey()
result,_ = detect_face(img)
#debug
img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
print(_)
cv2.imshow('image',img)
cv2.waitKey()