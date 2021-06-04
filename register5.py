import cv2
import face_detection
import face_recog
import os
import time
import pickle_man
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height
name = input("Enter your name: ")
img_path = f'registered_img/{name}/'
if not os.path.exists(img_path):
    os.makedirs(img_path)
count = 0

#capture images
while True:
    ret, frame = cap.read()
    max_width = cap.get(3) * 0.5
    max_height = cap.get(4) * 0.7
    left_offset = int (cap.get(3)*0.25)
    upper_offset = int (cap.get(4)*0.15)
    right_offset = int(left_offset+max_width)
    lower_offset = int(upper_offset+max_height)

    min_width = cap.get(3) * 0.3
    min_height = cap.get(4) * 0.4

    cv2.rectangle(frame, (left_offset, upper_offset), (right_offset,lower_offset), (255,0, 0), 2)
    if ret:
        face_list, boxes = face_detection.detect_face(frame)
        for face, box in zip(face_list, boxes):
            x,y,w,h = box
            if not (x>left_offset and x+w < right_offset and y > upper_offset and y+h < lower_offset):
                cv2.putText( frame, "move inside the box", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
                continue
            elif w < min_width and h < min_height:
                cv2.putText( frame, "move closer", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
                continue
            elif w > max_width and h > max_height:
                cv2.putText( frame, "move further", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
                continue
            else:
                img_save_path=img_path+'image'+str(count)+'.jpg'
                cv2.imwrite(img_save_path,face)
                time.sleep(2)
                print('image'+str(count)+' saved')
                count+=1
            
    cv2.putText(frame, f"capture done: {count}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('video', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27 or count == 5:  # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()

#extract feature
model = face_recog.RecognitionModel("model/model-4.h5")

img_list = [m for m in os.listdir(f"registered_img/{name}")]
feature_path = "registered_feature"
if not os.path.exists(feature_path):
    os.makedirs(feature_path)

extracted_feat = []
for img_name in img_list:
    img = cv2.imread(img_path+img_name)
    feat = model.extract(img)
    extracted_feat.append(feat)

extracted_feat = np.asarray(extracted_feat)
mean_feature = np.mean(extracted_feat.reshape(len(img_list),128), axis=0) 

#save feature in pickle format
database_path = "registered_feature/database.pkl"
pic_man = pickle_man.PickleMan(database_path)
pic_man.add(name, mean_feature)
pic_man.save_data(database_path)
print(f"{name} feature added to {database_path}")
