import cv2
import os
import numpy as np
import pickle

class datasetCreator():
    def __init__(self):
        self.DATADIR = "faceRecog/Dataset/"
        if not os.path.isdir(self.DATADIR):
            os.mkdir(self.DATADIR)
        self.face_cascade = cv2.CascadeClassifier('faceRecog/haarcascade_frontalface_default.xml')

    def makeDataset(self, img, ctr, name):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray,1.3,5)
        croppedImage = []
        for(x,y,w,h) in faces:
            croppedImage = gray[y:y+h, x:x+w]
            cv2.imwrite(self.DATADIR+name+"."+str(ctr)+".jpg",croppedImage)
            ctr+=1
        return ctr, croppedImage

class trainer():
    def __init__(self):
        self.DATADIR = "faceRecog/Dataset/"
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
    def train(self):
        x_train = []
        y_labels = []
        images = os.listdir(self.DATADIR)
        id = 0
        user_info = {}
        for image in images:
            inDict = False
            path = os.path.join(self.DATADIR, image)
            img = cv2.imread(path)
            print(path)
            print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            label = image[:image.index(".")]
            if not user_info:
                user_info[id] = label
            for key, value in user_info.items():
                if label == value:
                    id = key
                    inDict = True
                    break
                else:
                    inDict = False
            if not inDict:
                id = id+1
                user_info[id] = label
            x_train.append(img)
            y_labels.append(id)
        self.recognizer.train(x_train, np.array(y_labels))
        self.recognizer.save("faceRecog/face-trainner.yml")

        outFile = open("faceRecog/labelData.txt", "w")
        for i in user_info:
            outStr = str(i) + "," + user_info[i] + "\n"
            outFile.writelines(outStr)

        outFile.close()
        print(user_info)