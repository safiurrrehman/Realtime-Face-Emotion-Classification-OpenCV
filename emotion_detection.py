import cv2
import numpy as np
from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image

faceCascade= cv2.CascadeClassifier('/Users/safiurrehman/Desktop/Emotion Detection DL/haarcascade_frontalface_default.xml')
classifier =load_model('/Users/safiurrehman/Desktop/Emotion Detection DL/Emotion_saved.h5')

emotions = ['Angry','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0)



while True:
    ret, frame = cap.read()
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = img[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        pixel = image.img_to_array(roi_gray)
        pixel = np.expand_dims(pixel, axis = 0)  
        pixel /= 255  
        preds = classifier.predict(pixel)
        label = int(np.argmax(preds[0]))
        
        
        (Angry, Happy, Neutral, Sad, Surprise) = classifier.predict(pixel)[0]
        
        labels = "{}: {:.2f}%".format(emotions[label], max(Angry, Happy, Neutral, Sad, Surprise) * 100)
        cv2.putText(frame, labels, (x+20, y-60), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
