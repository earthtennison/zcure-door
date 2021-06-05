import numpy as np
import cv2
import os
import time
import tensorflow as tf
from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras import backend as K
import pickle


K.clear_session()
device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])





def preprocess(img):    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = img
    face = cv2.resize(face,(224,224))
    face=np.array((face,))
    return face



faceCascade = cv2.CascadeClassifier(
    'Cascades/haarcascade_frontalface_default.xml')
#eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
#smileCascade = cv2.CascadeClassifier('Cascades/haarcascade_smile.xml')
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height
name = input("Enter your name: ")
img_path = f'registered_img/{name}/'
if not os.path.exists(img_path):
    os.makedirs(img_path)
i = 0
while cap.isOpened():
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    a = int (cap.get(3)*0.3)
    b = int (cap.get(4)*0.15)
    cv2.rectangle(img, (a, b), (int(a+(cap.get(3) * 0.45)), int(b+(cap.get(4) * 0.7))), (0, 255, 0), 2)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x-2, y-2), (x + w+2, y + h+2), (255, 0, 0), 2)
        if a<x and x+w< int(a+(cap.get(3) * 0.45))and b<y and y+h<int(b+(cap.get(4) * 0.7)) and w > cap.get(3)*0.3 and h > cap.get(4)*0.4:
            cropped_face = img[y:y + h, x:x + w]
            img_save_path=img_path+'image'+str(i)+'.jpg'
            cv2.imwrite(img_save_path,cropped_face)
            time.sleep(2)
            print('image'+str(i)+' saved')
            i+=1
            
    cv2.putText(img, f"capture done: {i}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

 
        
    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27 or i ==5:  # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()


vgg_features = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling='avg')

img_list = [m for m in os.listdir(f"registered_img/{name}")]
feature_path = "registered_feature"
if not os.path.exists(feature_path):
    os.makedirs(feature_path)

extracted_feat = []
for img_name in img_list:
    img = cv2.imread(img_path+img_name)
    img = preprocess(img)
    #extract feature

    feature = vgg_features.predict(img)
    extracted_feat.append(feature)

extracted_feat = np.asarray(extracted_feat)
mean_feature = np.mean(extracted_feat.reshape(5,2048), axis=0)
print(mean_feature)
with open (feature_path+"/"+name+"_feature.pkl", 'wb') as f:
    pickle.dump(mean_feature, f)

print("Registered "+name)
