
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
import threading
import tensorflow as tf

label = "....."
n_timesteps = 10
lm_list=[] #luu gia tri cua khung xuong

model = tf.keras.models.load_model("model1.h5")

#doc anh tu webcam
cap = cv2.VideoCapture(0)

#khoi tao thu vien mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

def make_landmark_timestep(results):
    #print(results.pose_landmarks.landmark) #toa do cac diem tren khung xuong
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    #ve cac duong noi
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    #ve cac diem nut
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x+w), int(lm.y+h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED) #red1
    return img 

def draw_class_on_image(label, img): #gán nhãn lên ảnh
    font = cv2.FONT_HERSHEY_COMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0) #green1
    thickness = 2
    lineType = 2
    cv2.putText(img, label, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
    return img

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    #print(lm_list.shape)
    results=model.predict(lm_list)
    print("results:", results)
    if results[0][0] > 0.5:
         label = "body swing" #lắc người
    else: 
        label = "hand swing" #vẫy tay
    return label


i = 0
warmup_frames = 60

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #chuyen anh sang mau xam
    results=pose.process(imgRGB)
    i = i + 1
    if i > warmup_frames:
        print("Start detect...")

        if results.pose_landmarks:
            #ghi nhan thong so khung xuong
            c_lm = make_landmark_timestep(results)
            lm_list.append(c_lm) 

            if len(lm_list) == n_timesteps:
                #predict
                t1 = threading.Thread(target=detect, args=(model, lm_list,))
                t1.start()
                lm_list=[]
            #ve khung xuong len anh
            img =  draw_landmark_on_image(mpDraw, results, img)
        
        img = draw_class_on_image(label, img)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()