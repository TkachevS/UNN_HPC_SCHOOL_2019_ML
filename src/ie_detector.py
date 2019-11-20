"""

Inference engine detector
 
"""
import cv2
import math
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore
from time import time
#Labels of network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

class InferenceEngineDetector:
    def __init__(self, weightsPath = None, configPath = None,
                 device = 'CPU', extension = None):
        self.ie = IECore()
        if extension:
            self.ie.add_extension(extension, 'CPU')
        self.net = IENetwork(model=configPath, weights=weightsPath)
        self.exec_net = self.ie.load_network(network=self.net, device_name='CPU')

    def draw_detection(self, detections, img):    
        cols = img.shape[1] 
        rows = img.shape[0]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1]) 
                xLeftTop = int(detections[0, 0, i, 3] * cols) 
                yLeftTop = int(detections[0, 0, i, 4] * rows)
                xRightBottom = int(detections[0, 0, i, 5] * cols)
                yRightBottom = int(detections[0, 0, i, 6] * rows)
                cv2.rectangle(img, (xLeftTop, yLeftTop), (xRightBottom, yRightBottom), (0, 255, 0), 3)
                log.info(str(classNames[class_id]) + " conf.: " + str(confidence))
                cv2.putText(img, str(classNames[class_id]) + " " + str (round(confidence, 2)), (xLeftTop, yLeftTop), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
        return img

    def _prepare_image(self, image, h, w):
        image = cv2.resize(image, dsize = (w,h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1)) 
        image = np.expand_dims(image, axis = 0)
        return image
        
    def detect(self, image):
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))
        n, c, h, w = self.net.inputs[input_blob].shape
        blob = self._prepare_image(image, h, w)
        times = []
        t0_total = time()
        # for getting FPS 
        number_iter = 1
        for i in range(0,number_iter):
            t0 = time()
            output = self.exec_net.infer(inputs = {input_blob: blob})
            t1 = time()
            times.append(t1 - t0)
        t1_total = time()
        latency = np.median(times)
        fps = number_iter/(t1_total - t0_total)
        log.info("FPS: " + str(fps))
        log.info("Latency: " + str(latency))
        output = output[out_blob]
        detection = self.draw_detection(detections = output, img = image)        
        return detection