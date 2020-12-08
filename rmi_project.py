import numpy as np
import cv2
import urllib.request
import matplotlib.pyplot as plt
import tensorflow 
from tensorflow import keras
from tensorflow.keras.models import load_model
model=load_model(r'C:\Users\DIVYA A B\Downloads\letter_recognition_cnn.h5')
print(model.summary)


url='http://192.168.1.14:8080/shot.jpg'
while True:
    imgresp=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgresp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    cv2.imshow('test',img)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    elif key==ord('s'):
        cv2.imwrite("ocr.jpeg",img)


ocr=cv2.imread(r"C:\rmi\ocr.jpeg")
gray = cv2.cvtColor(ocr, cv2.COLOR_BGR2GRAY) 
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
cv2.imshow("thresh1",thresh1)
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("length",len(contours))
from imutils.contours import sort_contours
import imutils
contours = sort_contours(contours, method="left-to-right")[0]



  
processed_story = []
for cnt in contours: 
       x,y, w, h = cv2.boundingRect(cnt)
       if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
         if w > h:
             y=y-20
             x=x-20
             crop= thresh1[y:y+w+40, x:x+w+40]
         else:
             x =x-20
             y=y-20
             crop= thresh1[y:y+h+40, x:x+h+40]
     
         """rect = cv2.rectangle(thresh1, (x, y), (x + w, y + h), (0, 255, 0), 2) 
         cropped = thresh1[y:y + h+2, x:x + w+2]"""
         img = cv2.resize(crop, (28,28), interpolation = cv2.INTER_CUBIC)
         img = img/255
         img = img.reshape((28,28))
         processed_story.append(img)
         single_item_array = (np.array(img)).reshape(1,28,28,1)
         plt.imshow(img)
         plt.show()
         preds = model.predict(single_item_array)
         number=np.argmax(preds)
         print(chr(number+96))
         cv2.waitKey(2000)

cv2.waitKey(0)

cv2.destroyAllWindows()





    


