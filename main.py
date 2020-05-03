
#step one
import numpy as np
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("../data/lfw/*/*"))
dog_files = np.array(glob("../data/dog_images/*/*/*"))

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
#print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
#plt.show()

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


human_files_short_numbers=0
for H in range(len(human_files_short)):
    if face_detector(human_files_short[H]) is True:      
        human_files_short_numbers+=H
percentage_human_files_short=human_files_short_numbers/len(human_files_short)
print("percentage of the first 100 images in human_files is",percentage_human_files_short,"% have a detected human face")

dog_files_short_numbers=0
for D in range(len(dog_files_short)):
    if face_detector(dog_files_short[H]) is True:      
        dog_files_short_numbers+=H
percentage_dog_files_short=dog_files_short_numbers/len(dog_files_short)
print("percentage of the first 100 images in dog_files is",percentage_dog_files_short,"% have a detected human face")
## Step 2: Detect Dogs      
#Obtain Pre-trained VGG-16 Model

import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()
#(IMPLEMENTATION) Making Predictions with a Pre-trained Model
#a function that accepts a path to an image (such as 'dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg') as input and returns the index corresponding 
#to the ImageNet class that is predicted by the pre-trained VGG-16 model. 
#The output should always be an integer between 0 and 999,

from PIL import Image
import torchvision.transforms as transforms

# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    # 1: Input is an color image which is passed through convolution layers and pooling layers.
    
    image = Image.open(img_path).convert('RGB')
    #transform image to tensor to feed into the vgg16 model

    toTensor = transforms.ToTensor()
    #resize all images to 250 h and w
    transformation = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.ToTensor()])
    
    img_tensor = transformation(image)
    img_tensor = img_tensor.unsqueeze(0)

    # 2: Output of step 1 is a vector which is taken as an input for fully connected layer.
    in_transform = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    
    #img_tensor = in_transform(image)[:3,:,:].unsqueeze(0)
    img_tensor = in_transform(image)
    #if torch.cuda.is_available():
    img_tensor = img_tensor.cuda()
    
    prediction = VGG16(img_tensor.unsqueeze(0))
    
    #cpu processing
    #if torch.cuda.is_available():
    #    prediction = prediction.cpu()
    
    index = prediction.data.numpy().argmax()
    
    return index # predicted class index
 # predicted class index
    ### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    img_index = VGG16_predict(img_path)
    return (151 <= img_index and img_index <= 268) # true/false
human_files_short_numbers=0
for H in range(len(human_files_short)):
    if dog_detector(human_files_short[H]) is True:      
        human_files_short_numbers+=H
percentage_human_files_short=human_files_short_numbers/len(human_files_short)
print("percentage of the first 100 images in human_files is",percentage_human_files_short,"% have a detected dog face")
dog_files_short_numbers=0
for D in range(len(dog_files_short)):
    if dog_detector(dog_files_short[H]) is True:      
        dog_files_short_numbers+=H
percentage_dog_files_short=dog_files_short_numbers/len(dog_files_short)
print("percentage of the first 100 images in dog_files is",percentage_dog_files_short,"% have a detected dog face")


#Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
