import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd

class faceRecog():
    def __init__(self):
        self.fontface = cv2.FONT_HERSHEY_SIMPLEX
        self.fontscale = 1
        self.fontcolor = (255, 255, 255)
        self.userData = pd.read_pickle(r'faceRecog1/labelData.pickle')
        self.motionGraph = np.zeros([50, 1])
        self.face_cascade = cv2.CascadeClassifier('faceRecog1/haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("faceRecog1/face-trainner.yml")

    def detFaces(self, img):
        iter = 0
        Id = -1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            Id,conf = self.recognizer.predict(gray[y:y+h,x:x+w])
            cv2.putText(img,str(Id) + self.userData[Id],(x,y+h),self.fontface,self.fontscale,self.fontcolor)
        return Id, img
