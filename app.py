from tensorflow.keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
model = load_model("./Face_mask_detection_mobilenet.h5") 

classes = ["with_mask", "without_mask"]

while True:
    ret, frame = cam.read()
    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    
    for face in faces:

        x, y, w, h = face

        offset = 15
        face_section = frame[y-offset: y+h+offset, x-offset: x+w+offset]
        face_section = cv2.resize(face_section, (224, 224))
        a = image.img_to_array(face_section)
        a = np.expand_dims(a, axis=0)
        a = preprocess_input(a)
        value = np.argmax(model.predict(a),axis = 1)
        value = classes[value[0]]
        
        if(value == 'with_mask'):
            color = (0,255,0)
        else:
            color = (0,0,255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, value, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (225,0,0), 1, cv2.LINE_AA)
        # cv2.putText(frame, str(np.max(model.predict(a))), (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (225,0,0), 1, cv2.LINE_AA)
    
    cv2.imshow("window", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
