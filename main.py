
#step one
import numpy as np
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("E:/bicture/deepLearning/dog-classification/lfw/*/*"))
dog_files = np.array(glob("E:/bicture/deepLearning/dog-classification/dogImages/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))

import cv2                
import matplotlib.pyplot as plt                        

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]
#print(human_files_short)
#-#-# Do NOT modify the code above this line. #-#-#

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
    human_files_short_numbers=0
    for H in human_files_short:
        img = cv2.imread(human_files_short[H])
        # convert BGR image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find faces in image
        faces = face_cascade.detectMultiScale(gray)
        human_files_short_numbers+=len(faces)
    print("percentage of the first 100 Human images is"+human_files_short_numbers/len(human_files_short))
    dog_files_short_numbers=0
    for D in dog_files_short:
        img = cv2.imread(dog_files_short[D])
        # convert BGR image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find faces in image
        faces = face_cascade.detectMultiScale(gray)
        dog_files_short_numbers+=len(faces)
    print("percentage of the first 100 dog images is"+dog_files_short_numbers/len(dog_files_short))
### (Optional) 
### TODO: Test performance of anotherface detection algorithm.
### Feel free to use as many code cells as needed.
