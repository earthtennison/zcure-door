"""
The script is used for registered random person from vgg-face dataset and save it in pickle
"""
import cv2
import os
import urllib.request
import numpy as np
from numpy.core.defchararray import index
import face_detection
import random
import face_recog
import pickle

def url_to_img(url):
    try:
        resp = urllib.request.urlopen(url)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return img
    except:
        return None

PERSON_NUM = 100
IMAGE_PER_PERSON = 5
person_list = os.listdir("vgg_face_dataset/files")

sample_person_list = random.sample(person_list, PERSON_NUM)
print(sample_person_list)

face_recog.init()
print("model initiated")
feature_dict = {}

for person_file in sample_person_list:

    count = 0
    person_name = person_file[:-4]
    extracted_feat = []

    f = open(os.path.join("vgg_face_dataset/files",person_file), "r")
    for line in f:
        #check if the image is in curate dataset, more info in README.md
        if int(line.split(" ")[-1]) == 1:
            img = url_to_img(line.split(" ")[1])
            if img is None:
                continue
            #detect face
            cropped_img, (x,y,w,h) = face_detection.detect_face(img)
            if cropped_img is None:
                continue

            # using the vgg labels are not accurate
            # left, top, right, bottom = [int(float(f)) for f in line.split(" ")[2:6]]

            #display image
            # cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0),1)
            # cv2.imshow("img", img)
            # cv2.waitKey(1000)
            count += 1

            #extract feature
            feature = face_recog.extract(cropped_img)
            extracted_feat.append(feature)
            
        if count >= IMAGE_PER_PERSON:
            #wrap 5 features up to 1 representative feature
            extracted_feat = np.asarray(extracted_feat)
            mean_feature = np.mean(extracted_feat.reshape(IMAGE_PER_PERSON,2048), axis=0)
            #save it in feature dict
            feature_dict[person_name] = mean_feature
            print(f"{person_name} registered")
            break

with open(f"registered_feature/feature_{PERSON_NUM}.pkl","wb") as f:
    pickle.dump(feature_dict, f)
print("feature saved")
