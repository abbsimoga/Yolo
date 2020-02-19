import cv2 as cv
import numpy as np
import argparse
import os
from PIL import ImageGrab

confThreshold = 0.25
nmsThreshold = 0.4
inpWidth = 608
inpHeight = 608

folder= "yolocards"
classesFile = os.path.sep.join([folder, "yolo.names"]) 
classes = None

with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConf =  os.path.sep.join([folder, "yolo.cfg"]) 
modelWeights =  os.path.sep.join([folder, "yolo.weights"])  

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIDs = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:        
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2]* frameWidth)
                height = int(detection[3]*frameHeight )

                left = int(centerX - width/2)
                top = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes,confidences, confThreshold, nmsThreshold )
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)


def drawPred(classId, conf, left, top, right, bottom):
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf

    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    cv.putText(frame, label, (left,top), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

winName = 'OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, inpHeight, inpWidth)

while cv.waitKey(1) < 0:

    frameCap = ImageGrab.grab(bbox=(0, 0, inpWidth, inpHeight))
    frameNP = np.array(frameCap)
    frame = cv.cvtColor(frameNP, cv.COLOR_BGR2RGB)
 
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)

    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))

    postprocess (frame, outs)

    cv.imshow(winName, frame)

# cap = cv.VideoCapture(0)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, inpWidth)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, inpHeight)

# while cv.waitKey(1) < 0:

#     hasFrame, frame = cap.read()
#     frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

#     blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)

#     net.setInput(blob)
#     outs = net.forward (getOutputsNames(net))

#     postprocess (frame, outs)

#     cv.imshow(winName, frame)