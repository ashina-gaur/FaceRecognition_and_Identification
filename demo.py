import cv2
import numpy as np
import pickle
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels={"person_name":1}
with open("labels.pickle",'rb') as f: #read binary
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame

    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=img[y:y+h,x:x+w]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        id_,conf=recognizer.predict(roi_gray)
        if(conf >=45 and conf<=85):
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX 
            name=labels[id_]
            color=(255,255,255)
            stroke=4
            cv2.putText(img,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()
