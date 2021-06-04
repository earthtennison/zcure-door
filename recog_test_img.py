import cv2
import os
import face_detection
import face_recog
import pickle_man

person_list, features = pickle_man.get_data("registered_feature/feature_50.pkl")

img_path = "sample_image/brittany_curran2.jpg"
img = cv2.imread(img_path)
cropped_img, _ = face_detection.detect_face(img)

face_recog.init()
feat = face_recog.extract(cropped_img)
min_idx = face_recog.recognize(feat, features, thresh=200)
print(person_list[min_idx])
cv2.imshow("img",cropped_img)
cv2.waitKey(0)
