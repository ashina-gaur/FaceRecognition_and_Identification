import os
import cv2
import pickle
from PIL import Image
import numpy as np
x_train=[]
y_labels=[]
current_id=0
label_ids={}
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
basedir=os.path.dirname(os.path.abspath(__file__))
print(basedir)
# using the lbph face recogniser
recognizer=cv2.face.LBPHFaceRecognizer_create()

img_dir=os.path.join(basedir,"images")
print(img_dir)

for root,dirs,files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            
            path=os.path.join(root,file)
            label=os.path.basename(root)
            print(path)
            print(label)
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
            pil_image=Image.open(path).convert("L") #converting image to grayscale
            image_array=np.array(pil_image,"uint8")
            print(image_array)
            faces=face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for x,y,w,h in faces:
                roi= image_array[y: y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
print(y_labels)
print(x_train)
print(label_ids)

with open("labels.pickle",'wb') as f: #write binary
    pickle.dump(label_ids,f)
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")