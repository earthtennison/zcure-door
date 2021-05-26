from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
import cv2
import time
import numpy as np
import os
from keras import backend as K
from adafruit_servokit import ServoKit
import board
import busio
import time
import tensorflow as tf


K.clear_session()
device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
model_path = os.path.join("model",'model-1.h5')
feat_extractor = tf.keras.models.load_model(model_path)

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


def get_feat(img):
    feature = tf.math.l2_normalize(feat_extractor(img),axis=1)

def preprocess_input(x):
    x_temp = np.copy(x)

    x_temp = x_temp[..., ::-1]
    x_temp[..., 0] -= 93.5940 #subtract by mean of red channel
    x_temp[..., 1] -= 104.7624 #subtract by mean of green channel
    x_temp[..., 2] -= 129.1863 #subtract by mean of blue channel
 
    return x_temp


def preprocess_final(img):
    img = cv2.resize(img, (224,224))
    img = np.asarray(img, dtype=np.float64)
    img = preprocess_input(img)
    return img

img_list = [f for f in os.listdir("sample_image") if "regis" in f]
regis_data = {}
for img_name in img_list:
    img_path = os.path.join("sample_image",img_name)
    img = cv2.imread(img_path)
    img = preprocess_final(img)

    feature = get_feat(img)
    person_name = img_name.split("_")[0]
    print("Registered: "+person_name)
    regis_data[person_name] = feature

regis_person = sorted(list(regis_data.keys()))
extracted_feat = []
for person in regis_person:
    extracted_feat.append(regis_data[person])

img_test_path=os.path.join("sample_image","dene_infer")
img_test = cv2.imread(img_test_path)
img_test = preprocess_final(img_test)
feature_test = get_feat(img_test)
extracted_feat=np.asarray(extracted_feat)
distances = [np.linalg.norm(f) for f in extracted_feat-feature_test]
print(distances)