"""
The module is used for face detection
"""
import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
def detect_face(img):
    """
    :param img: cv2 image object or numpy array with channel last
    :type img:  numpy array

    :return: cropped face and position x, y, width, and height of cropped face in source image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    cropped_face = None
    (x0, y0, w0, h0) = (0,0,0,0)
    if len(faces) == 1:
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
            (x0, y0, w0, h0) = (x, y, w, h)

    return cropped_face, (x0, y0, w0, h0)

def detect_smile(img):
    pass

def detect_eye(img):
    pass
