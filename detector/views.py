from django.shortcuts import render, redirect
import cv2
from django.contrib.auth.forms import UserCreationForm
import threading
from django.http import StreamingHttpResponse
import RPi.GPIO as GPIO
import twilio
from twilio.rest import Client
import time
import datetime

from faceRecog.makeDataset import datasetCreator, trainer

from faceRecog.recognizer import faceRecog as FR
from objectRecog.recognizer import objectRecog as OR
from licenseRecog.recognizer import licenseRecog as LR

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(21,GPIO.IN)
GPIO.setup(26,GPIO.OUT)

DC = datasetCreator()
train = trainer()

FRO = FR()
ORO = OR()
LRO = LR()


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        return image

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()
            
cam = VideoCamera()

def homepage(request):
    return render(request, 'home.html')

def click(request):
    return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('/login')
    else:
        form  = UserCreationForm()
        args = {'form': form}
        return render(request, 'registrationForm.html', args)

def userHome(request):
    
    return render(request, 'userHome.html')

def mode1loop(camera):
    Id = -1
    ctr = 0
    frame_height = 500
    frame_width = 900
    sendSMS = False
    dt = datetime.datetime.now()
    minP = dt.minute
    vidName = 'vidLog/Face'+str(dt)+".avi"
    out = cv2.VideoWriter(vidName,cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
    makeNew = False
    screenAlert = "DANGER UNKNOWN PERSON"
    while True:
        extraInfo = ""
        faceLog = open('faces.log', 'a')
        frame = camera.get_frame()
        Id, name, frame = FRO.detFaces(frame)
        date = datetime.datetime.now()
        vidName = 'vidLog/Face'+str(date)+".avi"

        #print(ctr)
        if GPIO.input(21) ==False:
            if makeNew:
                out = cv2.VideoWriter(vidName,cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
                makeNew = False
            recordVid(out, frame)
        else:
            try:
                out.release()
                makeNew = True
            except:
                pass
        if Id<-5:
            ctr+=1
            frame = showAlert(frame, ctr, screenAlert)
            outStr = str(date) + ' : ' + "Unknown\n"
            faceLog.writelines(outStr)
        else:
            if Id>=0:
                outStr = str(date) + ' : ' + name +"\n"
                faceLog.writelines(outStr)
            ctr = 0
            resetAlarm()
        if ctr > 20:
            extraInfo = "Buzzer Activated:SMS Sent\n"
            raiseAlarm()
            faceLog.writelines(extraInfo)
            ctr = 0
            msg = "\nUnknown Person."
#             if not sendSMS:
            sendsms(msg)
            #sendSMS = True
        faceLog.close()
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame =  jpeg.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def mode1(request):
    global cam
    #cam = VideoCamera()
    try:
        return StreamingHttpResponse(mode1loop(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        pass
    
def mode2loop(camera):
    ctr = 0
    dt = datetime.datetime.now()
    minP = dt.minute
    vidName = 'vidLog/Object'+str(dt)+".avi"
    out = cv2.VideoWriter(vidName,cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
    sendSMS = False
    makeNew = False
    f = open('objectRecog/dangerousObjects.txt', 'r')
    dangObj = f.readlines()
    screenAlert = "DANGEROUS OBJECT DETECTED"
    extraInfo = ""
    while True:
        name = ""
        objectLog = open('objects.log', 'a')
        frame = camera.get_frame()
        objs, frame = ORO.detObjects(frame)
        date = datetime.datetime.now()
        vidName = 'vidLog/Object'+str(date)+".avi"
        if GPIO.input(21) ==False:
            if makeNew:
                out = cv2.VideoWriter(vidName,cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
                makeNew = False
            recordVid(out, frame)
        else:
            try:
                out.release()
                makeNew = True
            except:
                pass
        det = True
        for o in objs:
            for do in dangObj:
                if o.upper() == do[:-1].upper():
                    #msg = "\nUnknown Car, License Plate Number: \n"+o
                    frame = showAlert(frame, ctr, screenAlert)
                    ctr +=1
                    det = True
                    name = o
                    break
            if det:
                break
        print(ctr) 
        if ctr>5:
            msg = "\nDangerous Object Detected: \n"+name
            raiseAlarm()
            #if not sendSMS:
            sendsms(msg)
            ctr = 0
            #sendSMS = True
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame =  jpeg.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        for o in objs:
            if o == name:
                extraInfo = "Dangerous Object:Buzzer Activated:SMS Sent"
            else:
                extraInfo = ""
            opStr = str(date)+":"+o+":"+extraInfo+"\n"
            objectLog.writelines(opStr)
        objectLog.close()
    f.close()

def mode2(request):
    #cam = VideoCamera()
    try:
        #return render(request, 'home.html')
        return StreamingHttpResponse(mode2loop(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        pass

def mode3loop(camera):
    f = open('licenseRecog/allowed.txt', 'r')
    allowed = f.readlines()
    allowCar = False
    dt = datetime.datetime.now()
    minP = dt.minute
    vidName = 'vidLog/License'+str(dt)+".avi"
    out = cv2.VideoWriter(vidName,cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
    sendSMS = True
    makeNew = False
    while True:
        licenseLog = open('license.log', 'a')
        frame = camera.get_frame()
        name = LRO.detLicense(frame)
        date = datetime.datetime.now()
        vidName = 'vidLog/License'+str(date)+".avi"
        screenAlert = "UNKNOWN CAR DETECTED"
        carState = ""
        extraInfo = ""
        if GPIO.input(21) ==False:
            if makeNew:
                out = cv2.VideoWriter(vidName,cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
                makeNew = False
            recordVid(out, frame)
        else:
            try:
                out.release()
                makeNew = True
            except:
                pass
        for plate in allowed:
            if plate[:-1].upper()  == name.upper() or len(name)==0:
                allowCar = True
                carState = "KNOWN"
                extraInfo = ""
                break
            else:
                allowCar = False
        if not allowCar:
            carState = "UNKNOWN"
            extraInfo = "Buzzer Activated:SMS Sent"
            msg = "\nUnknown Car, License Plate Number: \n"+name
            raiseAlarm()
            frame = showAlert(frame, 0, screenAlert)
            if sendSMS:
                sendsms(msg)
                sendSMS = False
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame =  jpeg.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        if len(name)>0:
            opStr = str(date)+":" + carState + ":" + name + ":" + extraInfo+"\n"
            licenseLog.writelines(opStr)
        licenseLog.close()
    f.close()

def mode3(request):
    #cam = VideoCamera()
    try:
        #return render(request, 'home.html')
        return StreamingHttpResponse(mode3loop(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        pass
    
def addLicense(request):
    license = input("Enter License Plate Number: ")
    license = license + "\n"
    file = open("licenseRecog/allowed.txt", "a")
    file.writelines(license)
    file.close()
    return redirect('/login/userhome/')

def takePictures(request):
    return render(request, 'adduserimages.html')

def addFace(request):
    #cam = VideoCamera()
    i = 0
    name = input('Input user name: ')
    
    while True:
        frame = cam.get_frame()
        i, face = DC.makeDataset(frame, i, name)
        print(i)
        if len(face)>0:
            cv2.imshow('T', face)
            cv2.waitKey(30)
        if i > 20:#Number of images
            cv2.destroyAllWindows()
            break
    return redirect('/login/userhome/takepictures/')

def done(request):
    train.train()
    return redirect('/login/userhome/')

def raiseAlarm():
    for i in range(0, 5):
        GPIO.output(26, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(26, GPIO.LOW)
        time.sleep(0.1)
    
def resetAlarm():
    GPIO.output(26, GPIO.LOW)
    
def sendsms(msg):
    account_sid = ''
    auth_token = ''
    client = Client(account_sid, auth_token)

    message = client.messages.create(
             body='ALERT - Intrusion Detected ' + msg,
             from_='',
             to='')

    print(message.sid)
    
def showAlert(img, ctr, msg):
    try:
        R, C, _ = img.shape
    except:
        R, C = img.shape
    if ctr %2 ==0 :
        cv2.putText(img,msg,(int(20),int(30)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    return img
def recordVid(writer, frame):
    writer.write(frame)

    