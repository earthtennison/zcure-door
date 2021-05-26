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


vgg_features = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling='avg')


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

def preprocess_video(img):
    face, (x, y, w, h) = detect_face(img)
    face = cv2.resize(face,(224,224))
    face = np.array((face,))
    return face, (x, y, w, h)
    
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

#Registered person
img_list = [f for f in os.listdir("sample_image") if "regis" in f]
regis_data = {}
for img_name in img_list:
    img_path = os.path.join("sample_image",img_name)
    img = preprocess(img_path)
    #extract feature

    feature = vgg_features.predict(img)
    
    #person name
    person_name = img_name.split("_")[0]
    print("Registered: "+person_name)
    regis_data[person_name] = feature
regis_person = sorted(list(regis_data.keys()))
extracted_feat = []
for person in regis_person:
    extracted_feat.append(regis_data[person])

#Servo
i2c_bus0 = (busio.I2C(board.SCL_1,board.SDA_1))
kit = ServoKit(channels=16,i2c=i2c_bus0)






cap = cv2.VideoCapture(0)
_,frame = cap.read()
input_size = (frame.shape[1],frame.shape[0])
thresh = 100
print("Entering while loop")
while True:
    
    ret, frame = cap.read()
    
    if ret:
        frame = cv2.resize(frame, input_size)
        img_infer, (x,y,w,h) = preprocess_video(frame)
        if((x, y, w, h)==(0,0,0,0)):
            pass
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            feature_infer = vgg_features.predict(img_infer)
            extracted_feat = np.asarray(extracted_feat)
            distances = [np.linalg.norm(f) for f in extracted_feat-feature_infer]
            print(distances)
            if(np.min(distances)>thresh):
                print("Unauthorized")

            else:
                name = regis_person[np.argmin(distances)]
                cv2.putText( frame, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                print("Authorized")
                print("Opening Door")
                kit.servo[0].angle=90
                print("Door Opened")
                time.sleep(3)
                print("Closing Door")
                kit.servo[0].angle=0
                print("Door Closed")
                time.sleep(3)
        cv2.imshow('video',frame)
        cv2.waitKey(1)      
    else:
        cap.release()
        break
    

    
