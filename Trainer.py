import cv2, os
import numpy as np
from PIL import Image
import os



cascadePath = "/home/pi/Downloads/face_recognizer/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.face.createEigenFaceRecognizer()



def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        img = Image.open(image_path).convert('L')
        image = np.asarray(img)
        image = cv2.resize(image,(240,240))
        nbr = os.path.split(image_path)[1].split(".")[0]
        #nbr = int(filter(str.isdigit, nbr))
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image)
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image)
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels


path = 'myfaces'
images, labels = get_images_and_labels(path)
print labels


cv2.destroyAllWindows()
pics= []
for label in labels:
    if (label == "James"):
        pics.append(1)


np.savez_compressed('/tmp/', a =images, b = pics)
#np.save('/home/pi/Downloads/face_recognizer/Compressed/labels', labels)
print ("Done")

