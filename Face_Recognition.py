import cv2, os
import numpy as np
from PIL import Image
import os
from picamera.array import PiRGBArray
from picamera import PiCamera
import time


cascadePath = "/home/pi/Downloads/face_recognizer/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.face.createEigenFaceRecognizer()
path = "/home/pi/Downloads/face_recognizer/myfaces/"
directory = "/home/pi/Downloads/face_recognizer/"  
#framerate = 10
loop=True

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(320, 240))

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
        image = cv2.resize(image,(240,240))
        nbr = os.path.split(image_path)[1].split(".")[0]
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image)
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image)
            cv2.waitKey(50)
    # return the images list and labels list
    labelss= []
    for label in labels:
        if (label == "James"):
            labelss.append(1)
        else:
            labelss.append(0)
    cv2.destroyAllWindows()

    return images, labelss



def take_pic(name):
    print ("Enter anything to prepare the frame.")
    answer = raw_input()
    name = name + ".jpg"
    camera.start_preview()
    time.sleep(2)
    camera.capture(name)     #It saves in this directory
    camera.stop_preview()
    show = directory + name 
    show = cv2.imread(show, 0)
    cv2.imshow('Image',show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_face():
    answer = "N"
    print ("What is the name of the person you want to add?")
    name = str(raw_input())
    print ("You will take 3 pictures of the face you want to add: normal, surprised and happy")
    print ("Are you ready for the first picture...normal")
    while (answer == "N"): 
        take_pic(name)
        print ("Are you satisfied with this picture? Y foryes/ N for no")
        answer = str(raw_input())
    show = cv2.imread(directory + name + ".jpg", 0)
    cv2.imwrite(os.path.join(path,name + "1.jpg"), show)
    print ("Are you ready for the second picture...surprised")
    answer = "N"
    while (answer == "N"): 
        take_pic(name)
        print ("Are you satisfied with this picture? Y foryes/ N for no")
        answer = str(raw_input())
    show = cv2.imread(directory + name + ".jpg", 0)
    cv2.imwrite(os.path.join(path,name + "2.jpg"), show)
    print ("Are you ready for the third picture...happy?")
    answer = "N"
    while (answer == "N"): 
        take_pic(name)
        print ("Are you satisfied with this picture? Y foryes/ N for no")
        answer = str(raw_input())
    show = cv2.imread(directory + name + ".jpg", 0)
    cv2.imwrite(os.path.join(path,name + "3.jpg"), show)
    print ("Finished Adding Face!")



def remove_face():
    print ("Type the name of the face you want to remove")
    name = str(raw_input())
    if (os.path.isfile(path + name + "1.jpg")):
        os.remove(path + name + "1.jpg")
        os.remove(path + name + "2.jpg")
        os.remove(path + name + "3.jpg")
        print ("Faces Removed")
    else:
        print ("This face is not in the current directory")

    
def print_menu():       
    print (30 * "-" , "MENU" , 30 * "-")
    print ("1. Add Faces to Memory")
    print ("2. Remove Faces from Memory")
    print ("3. Change Resolution or Frame Rate   (Soon to add hopefully)")
    print ("4. Start Program")
    print ("5. Exit")
    print (67 * "-")
  
#def framerate():
#    print ("What would you like to change the framerate to?")
#    rate = str(raw_input())
#    framerate = rate

    
def camera_start():
    path = 'myfaces'
    images, labels = get_images_and_labels(path)
    recognizer.train(images, np.array(labels))
    print ("Ready to use Camera!")


    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                image = np.array(frame.array, dtype=np.uint8)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image,(240,240))


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


            
while loop:         
    print_menu()   
    choice = input("Enter your choice [1-5]: ")
     
    if choice==1:     
        print ("Menu 1 has been selected")
        add_face()
    elif choice==2:
        print ("Menu 2 has been selected")
        remove_face()
    #elif choice==3:
    #    print ("Menu 3 has been selected")
    #    framerate()
    elif choice==4:
        print ("Menu 4 has been selected")
        camera_start()
    elif choice==5:
        print ("System Exit")
        loop=False # This will make the while loop to end as not value of loop is set to False
    else:
        # Any integer inputs other than values 1-5 we print an error message
        raw_input("Wrong option selecti`	
		on. Enter any key to try again..")










