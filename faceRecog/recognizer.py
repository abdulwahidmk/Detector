import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd

class faceRecog():
    def __init__(self):
        self.fontface = cv2.FONT_HERSHEY_SIMPLEX
        self.fontscale = 1
        self.fontcolor = (0, 255, 0) #BGR
        self.motionGraph = np.zeros([50, 1])
        self.face_cascade = cv2.CascadeClassifier('faceRecog/haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("faceRecog/face-trainner.yml")

    def detFaces(self, img):
        iter = 0
        Id = -1
        inFile = open(r'faceRecog/labelData.txt', 'r')
        self.userData = {}
        name = ''
        for line in inFile:
            ind = line.find(',')
            index = int(line[:ind])
            name = line[ind+1:-1]
            self.userData[index] = name
            #print(index)
        inFile.close()
        #print(self.userData)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            Id,conf = self.recognizer.predict(gray[y:y+h,x:x+w])
            if conf>65: #Change this value to changethe threshold at witch the face should be detected.
                #If you make it high the detection criterio will be very lose. If you make it low the criterio will be strict
                Id = -10
            else:
                name = self.userData[Id]
                cv2.putText(img,str(Id) +" "+ self.userData[Id],(x,y+h),self.fontface,self.fontscale,self.fontcolor, 2)
        return Id, name, img
