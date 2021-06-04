"""
The module is used for face detection
"""
import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
def detect_face(img, scale = 1.3):
    """
    :param img: cv2 image object or numpy array with channel last
    :type img:  numpy array

    :return: list of cropped faces and list of boxes containing x, y, width, and height of cropped face in source image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    cropped_faces = []
    boxes = []
    if len(faces) >= 1:
        for (x, y, w, h) in faces:
            new_w, new_h = int(w*scale), int(h*scale)
            y = max(y + h//2 - new_h//2, 0)
            x = max(x + w//2- new_w//2, 0)
            cropped_face = img[y:y + new_h, x:x + new_w]
            cropped_faces.append(cropped_face)
            boxes.append((x, y, new_w, new_h))
    
    return cropped_faces, boxes

def detect_smile(img):
    pass

def detect_eye(img):
    pass
