from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from pymouse import PyMouse
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

m = PyMouse()


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 3

LEFTCOUNTER = 0
RIGHTCOUNTER = 0
BOTH = 0
LEFTTOTAL = 0
RIGHTTOTAL = 0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

while True:
    if fileStream and not vs.more():
        break
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if leftEAR < EYE_AR_THRESH and rightEAR < EYE_AR_THRESH:
            LEFTCOUNTER += 1
            RIGHTCOUNTER += 1
        elif leftEAR < EYE_AR_THRESH:
            LEFTCOUNTER += 1
        elif rightEAR < EYE_AR_THRESH:
            RIGHTCOUNTER += 1
        else:
            if LEFTCOUNTER >= EYE_AR_CONSEC_FRAMES and RIGHTCOUNTER >= EYE_AR_CONSEC_FRAMES:
                BOTH += 1
                x, y = m.position()
                m.click(x, y, 3)
            elif LEFTCOUNTER >= EYE_AR_CONSEC_FRAMES:
                LEFTTOTAL += 1
                x, y = m.position()
                m.click(x, y, 1)
            elif RIGHTCOUNTER >= EYE_AR_CONSEC_FRAMES:
                RIGHTTOTAL += 1
                x, y = m.position()
                m.click(x, y, 2)

            LEFTCOUNTER = 0
            RIGHTCOUNTER = 0

        cv2.putText(frame, "Blinks: {}".format(BOTH), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Left_Blinks: {}".format(LEFTTOTAL), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Right_Blinks: {}".format(RIGHTTOTAL), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "LEFT_EAR: {:.2f}".format(leftEAR), (250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "RIGHT_EAR: {:.2f}".format(rightEAR), (250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
