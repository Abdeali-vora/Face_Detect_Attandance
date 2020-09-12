
import numpy as np
import os
import face_recognition
import cv2
from datetime import datetime

path ='images'
images =[]
classnames = []
MyList = os.listdir(path)
for cl in MyList:
    cur_img = cv2.imread(f'{path}/{cl}')
    images.append(cur_img)
    classnames.append(os.path.splitext(cl)[0])

def Encodings(images):
    encodeList =[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def Attandance(name):
    with open('Attandance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            datestring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{datestring}')

encodeListKnown = Encodings(images)
cap = cv2.VideoCapture(0)

while True:
    successful_frame_read ,frame = cap.read()
    imgS = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    Face_cur_frame = face_recognition.face_locations(imgS)
    encode_cur_frame = face_recognition.face_encodings(imgS,Face_cur_frame)

    for encodeFace,FaceLoc in zip(encode_cur_frame,Face_cur_frame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

        matchindex = np.argmin(faceDis)

        if matches[matchindex]:
            name =classnames[matchindex].upper()
            
            y1,x2,y2,x1= FaceLoc
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
            Attandance(name)

    cv2.imshow('Hey You Look Beautiful Keep Watch In Camera',frame)  
    key = cv2.waitKey(1)
    if key ==65 or key ==97:
        break
    

