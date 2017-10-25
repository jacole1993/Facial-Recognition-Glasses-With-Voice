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

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.face.createEigenFaceRecognizer()

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        img = Image.open(image_path).convert('L')
        image = np.asarray(img)
        image = cv2.resize(image,(120,120))
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image)
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image)
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels



path = 'yalefaces'
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()


recognizer.train(images, np.array(labels))


image_paths = [os.path.join(path, f) for f in os.listdir(path)]

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = np.array(frame.array, dtype=np.uint8)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image,(120,120))

            #predict_image_pil = Image.open(image).convert('L')
            #predict_image = np.array(predict_image_pil, 'uint8')
            faces = faceCascade.detectMultiScale(image)
            for (x, y, w, h) in faces:
                nbr_predicted, conf = recognizer.predict(image)
                if nbr_predicted == 16:
                    os.system('flite -t "hello james"')
                print nbr_predicted, conf 

            cv2.imshow("Video", image)
            key = cv2.waitKey(1) # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break    

            rawCapture.truncate(0)
    

