from __future__ import absolute_import, division, print_function

import io
import json
import time

import requests
from PIL import Image

import cv2

class licenseRecog():
    def __init__(self):
        pass

    def detLicense(self, frame):
        result = []
        name = ""
        imgByteArr = io.BytesIO()
        im = Image.fromarray(frame)
        im.save(imgByteArr, 'JPEG')
        imgByteArr.seek(0)
        response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            files=dict(upload=imgByteArr),
            headers={'Authorization': 'Token '})
        result.append(response.json())
        try:
            name = response.json()['results'][0]['plate']
            print(name)
        except:
            name = ""
            pass
        time.sleep(1)
        return name