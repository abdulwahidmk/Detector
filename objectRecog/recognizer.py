import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util


class objectRecog():
    def __init__(self):
        self.MODEL_NAME = 'objectRecog/model'
        self.GRAPH_NAME = 'detect.tflite'
        self.LABELMAP_NAME = 'labelmap.txt'
        self.min_conf_threshold = 0.5
        self.resW, self.resH = '1280x720'.split('x')
        self.imW, self.imH = int(self.resW), int(self.resH)
        self.use_TPU = False

        self.pkg = importlib.util.find_spec('tflite_runtime')
        if self.pkg:
            from tflite_runtime.interpreter import Interpreter
            if self.use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if self.use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        if self.use_TPU:
            if (self.GRAPH_NAME == 'detect.tflite'):
                self.GRAPH_NAME = 'edgetpu.tflite'

        CWD_PATH = os.getcwd()

        PATH_TO_CKPT = os.path.join(CWD_PATH,self.MODEL_NAME,self.GRAPH_NAME)

        PATH_TO_LABELS = os.path.join(CWD_PATH,self.MODEL_NAME,self.LABELMAP_NAME)
        
        print(PATH_TO_LABELS)

        with open(PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        if self.labels[0] == '???':
            del(self.labels[0])

        if self.use_TPU:
            self.interpreter = s=Interpreter(model_path=PATH_TO_CKPT,
                                      experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(PATH_TO_CKPT)
        else:
            self.interpreter = Interpreter(model_path=PATH_TO_CKPT)

        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

    def detObjects(self, frame1):
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
        self.interpreter.invoke()

        # Retrieve detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects
        detectedObjects = []
        for i in range(len(scores)):
            if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * self.imH)))
                xmin = int(max(1,(boxes[i][1] * self.imW)))
                ymax = int(min(self.imH,(boxes[i][2] * self.imH)))
                xmax = int(min(self.imW,(boxes[i][3] * self.imW)))

                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                detectedObjects.append(object_name)
        return  detectedObjects, frame
