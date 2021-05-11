import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier(
    'Cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('Cascades/haarcascade_smile.xml')
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height
while cap.isOpened():
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for (x, y, w, h) in faces:
        a = int (cap.get(3)*0.3)
        b = int (cap.get(4)*0.15)
        cv2.rectangle(img, (a, b), (int(a+(cap.get(3) * 0.45)), int(b+(cap.get(4) * 0.7))), (0, 255, 0), 2)
        if w > cap.get(3)*0.3 and h > cap.get(4)*0.4:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            print(faces)
            print(1)
        else:
            print(0)
        roi_color = img[y:y + h, x:x + w]
        roi_gray = gray[y:y + h, x:x + w]

        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=10,
            minSize=(5, 5),
        )
        smiles = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25, 25),
        )

        # display face at top left
        roi_color = cv2.resize(roi_color, (50, 50))
        img[10:60, 10:60, :] = roi_color

        for eye in eyes:
            (ex, ey, ew, eh) = eye
            cv2.rectangle(img, (ex + x, ey + y),
                          (ex + ew + x, ey + eh + y), (0, 255, 0), 2)

        if len(smiles) != 0:
            # use the smallest box
            compare = []
            for smile in smiles:
                sw, sh = smile[2], smile[3]
                compare.append(sw * sh)
            max_ind = np.argmax(np.array(compare))

            (sx, sy, sw, sh) = smiles[max_ind]
            cv2.rectangle(img, (sx + x, sy + y),
                          (sx + sw + x, sy + sh + y), (0, 0, 255), 2)
    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
