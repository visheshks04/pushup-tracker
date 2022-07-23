import cv2 as cv
import mediapipe as mp
import time
import numpy as np

def y_dist(l1, l2):
    return abs(l1.y - l2.y)

cap = cv.VideoCapture(0) 

cam_width, cam_height = 1280, 800
cap.set(3, cam_width)
cap.set(4, cam_height)


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_draw = mp.solutions.drawing_utils

previous_time = 0

pushups_count = 0
lock = False

while True:

    success, img = cap.read()
    img_RGB = cv.cvtColor(img,  cv.COLOR_BGR2RGB)
    results = pose.process(img_RGB)

    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    left_side = y_dist(results.pose_landmarks.landmark[15], results.pose_landmarks.landmark[11])
    right_side = y_dist(results.pose_landmarks.landmark[16], results.pose_landmarks.landmark[12])

    print(left_side)
    print(right_side)

    if (not lock) and left_side < 0.2 and right_side < 0.2:
        pushups_count += 1
        lock = True

    if lock and left_side > 0.35 and right_side > 0.35:
        lock = False

    current_time = time.time()
    fps = 1/(current_time-previous_time) # Calculating Frames per second rate
    previous_time = current_time

    cv.putText(img, f'FPS: {int(fps)}', (10,30), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
    cv.putText(img, f'Pushups: {pushups_count}', (10,40), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)

    cv.imshow('Image', img)
    cv.waitKey(1)