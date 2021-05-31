"""
This script is used to test the performance of recognition process when threre are more person registered.
- Experiment on 10, 20, 30, 40, 50, 100 persons
- example person is collected from vgg_face_dataset
- Metric is accuracy (number of correction prediction / all registrored persons)
"""

import numpy as np
import cv2
import numpy as np
import face_detection
import face_recog
import os
import pickle_man
import urllib
import random

def url_to_img(url):
    try:
        resp = urllib.request.urlopen(url)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return img
    except:
        return None

#get registered data
pkl_path = "registered_feature/feature_50.pkl"
person_list, regis_feat = pickle_man.get_data(pkl_path)
print(person_list)

#get test images
face_recog.init()
correct = 0
for i in range(len(person_list)):
    f = open(os.path.join("vgg_face_dataset/files",person_list[i]+".txt"), "r")
    lines = f.readlines()
    print("-"*10)
    print(f"recognizing {person_list[i]}")
    while True:
        line = lines[random.randint(0,len(lines))]
        
        if int(line.split(" ")[-1]) == 1:
            img = url_to_img(line.split(" ")[1])
        else:
            continue

        if img is not None:
            cropped_img, _ = face_detection.detect_face(img)
            if cropped_img is not None:
                print(line.split(" ")[1])
                break

    cv2.imshow("img",cropped_img)
    
    feat = face_recog.extract(cropped_img)
    min_idx = face_recog.recognize(feat, regis_feat, thresh=200)    
    if min_idx is not None:
        print(f"predict : {person_list[min_idx]}")
        if min_idx == i:
            correct += 1
        
    else:
        print("predict : Unknown")
    
    cv2.waitKey(100)

print(f"accuracy {correct/len(person_list)}")

