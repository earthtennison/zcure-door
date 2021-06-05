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
import pickle

def url_to_img(url):
    try:
        resp = urllib.request.urlopen(url)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return img
    except:
        return None

#get registered data
pkl_path = "registered_feature/feature_trained5_50.pkl"
picman = pickle_man.PickleMan(pkl_path)
person_list, regis_feat = picman.get_data()
print(person_list)

#get test images
# model path "model/model-3.h5"
model = face_recog.RecognitionModel("model/model-5.h5")
print("model initialized")
correct = 0
min_distances = []
distance_debug = []
for i in range(len(person_list)):
    f = open(os.path.join("vgg_face_dataset/files",person_list[i]+".txt"), "r")
    lines = f.readlines()
    print("-"*10)
    print(f"recognizing {person_list[i]}")
    cropped_img = None
    while True:
        line = lines[random.randint(0,len(lines)-1)]
        
        if int(line.split(" ")[-1]) == 1:
            img = url_to_img(line.split(" ")[1])
        else:
            continue

        if img is not None:
            cropped_images, _ = face_detection.detect_face(img)
            if len(cropped_images) == 1:
                print(line.split(" ")[1])
                cropped_img = cropped_images[0]
                break

    cv2.imshow("img",cropped_img)
    
    feat = model.extract(cropped_img)
    min_idx, min_dis, distances = model.recognize(feat, regis_feat, thresh=200, debug=True)    
    if min_idx is not None:
        print(f"predict : {person_list[min_idx]}")
        if min_idx == i:
            min_distances.append(min_dis)
            distance_debug.append(distances)
            correct += 1
        else:
            print("! "*15+"WRONG"+"! "*15)
        
    else:
        print("predict : Unknown")
    
    cv2.waitKey(100)

#save debug
save_path = "registered_feature/distance_trained5_50.pkl"
with open(save_path,"wb") as f:
    pickle.dump(distance_debug, f)
print(f"pickle saved to {save_path}")

print(f"recognizing {len(person_list)} with accuracy {correct/len(person_list)}")
print(f"min distances of correct prediction:\n{min_distances}")

