# import the necessary packages
import numpy as np

import time
import cv2
import os

def get_detections(path_to_yolo, frame):
    min_confidence = 0.5 # 0.6
    min_threshold = 0.2 # 0.3

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([path_to_yolo, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # derive the paths to the YOLO weights and model configuration
    # weightsPath = os.path.sep.join([path_to_yolo, "yolov3.weights"])
    # configPath = os.path.sep.join([path_to_yolo, "yolov3.cfg"])
    weightsPath = os.path.sep.join([path_to_yolo, "yolov3-spp.weights"])
    configPath = os.path.sep.join([path_to_yolo, "yolov3-spp.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # loop over frames from the video file stream
    while True:
        (H, W) = frame.shape[: 2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

# TODO распараллелить
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                if classID != 0:
                    continue
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > min_confidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0: 4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence,
                                min_threshold)

        # ensure at least one detection exists
        detections = []
        out_scores = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            # TODO распараллелить сбор данных о bbox
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                detections.append([x, y, w, h])
                out_scores.append(confidences[i])
        if len(detections) == 0:
            detections = None
        return detections, out_scores
