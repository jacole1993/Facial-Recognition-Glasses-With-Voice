import cv2, os
import numpy as np
from PIL import Image
from picamera.array import PiRGBArray
from picamera import PiCamera
import os


camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(320, 240))

cascadePath = "/home/pi/Downloads/face_recognizer/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

recognizer = cv2.face.createEigenFaceRecognizer()


load = np.load('/tmp/.npz')
print load['b']

images = load['a']
labels = load['b']
recognizer.train(images, np.array(labels))
print ("Finished")
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = np.array(frame.array, dtype=np.uint8)
            view = image
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image,(240,240))

            #predict_image_pil = Image.open(image).convert('L')
            #predict_image = np.array(predict_image_pil, 'uint8')
            faces = faceCascade.detectMultiScale(image)
            for (x, y, w, h) in faces:
                nbr_predicted, conf = recognizer.predict(image)
                if nbr_predicted == 1:
                    os.system('flite -t "hello james"')
                    print ("James")
                

            cv2.imshow("Video", view)
            key = cv2.waitKey(1) # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break    

            rawCapture.truncate(0)




