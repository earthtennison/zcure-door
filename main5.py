from threading import main_thread
import cv2
from numpy.lib.function_base import extract
import pickle_man
import face_recog
import face_detection
import time
#get registered data
pkl_path = "registered_feature/database.pkl"
picman = pickle_man.PickleMan(pkl_path)
person_list, regis_feat = picman.get_data()
print("\nperon registered")
print(person_list)

model = face_recog.RecognitionModel("model/model-4.h5")

cap = cv2.VideoCapture(0)
_,frame = cap.read()
input_size = (frame.shape[1],frame.shape[0])
thresh = 40
start = 0

print("Entering while loop")
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
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                feat = model.extract(face)
                min_idx = model.recognize(feat, regis_feat, thresh=thresh)
                if min_idx is not None:
                    name = person_list[min_idx]
                    cv2.putText( frame, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    print(f"Welcome {name}")
                    start = time.time()
                    print("Door Open")
                else:
                    name = "unknown"
                    cv2.putText( frame, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                break
        if time.time() - start >= 3:
            print("Door Close")
        print('_'*10)
        cv2.imshow('video',frame)
        cv2.waitKey(1) 

    else:
        cap.release()
        break
    

