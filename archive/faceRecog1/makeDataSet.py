import face_recognition
import cv2
import os
import numpy as np

DATADIR = "Dataset/"

if not os.path.isdir(DATADIR):
    os.mkdir(DATADIR)
person = "zain"

def saveImg(img, ctr):
    cv2.imwrite(DATADIR+person+str(ctr)+".jpg", img)
    a = 1

def drawboxes(Loc, frame, i, ctr):
    top = Loc[i][0]
    right = Loc[i][1]
    bottom = Loc[i][2]
    left = Loc[i][3]
    crop_img = frame[top:bottom, left:right]
    LM = face_recognition.api.face_landmarks(crop_img, face_locations=None, model='small')
    # print(len(LM))
    if len(LM) != 0:
        saveImg(crop_img, ctr)
        cv2.imshow('face', crop_img)
        ctr = ctr + 1
    return ctr

cap = cv2.VideoCapture(0)
ctr = 0
while(ctr != 20):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Loc = face_recognition.face_locations(frame, number_of_times_to_upsample=1, model='hog')
    faces = len(Loc)
    if faces != 0:
        for i in range(0, faces):
            ctr = drawboxes(Loc, frame, i, ctr)
    # cv2.imshow('frame',frame)
    print(ctr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
