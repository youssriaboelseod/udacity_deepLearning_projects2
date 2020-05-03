
##step1
## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
    human_files_short_numbers=0
    for H in human_files_short:
        img = cv2.imread(human_files_short[H])
        # convert BGR image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find faces in image
        faces = face_cascade.detectMultiScale(gray)
        human_files_short_num
import torch
import torchvision.models as models
# define VGG16 model
VGG16 = models.vgg16(pretrained=True)
# check if CUDA is available
use_cuda = torch.cuda.is_available()
# move model to GPU if CUDA is available
if use_cuda:
print (use_cuda)
VGG16 = VGG16.cuda()
from PIL import Image
import torchvision.transforms as transforms
#Not sure this is needed, but was trying to troubleshoot the issue
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def VGG16_predict(img_path):
'''
Use pre-trained VGG-16 model to obtain index corresponding to
predicted ImageNet class for image at specified pat
Args:
img_path: path to an image
Returns:
Index corresponding to VGG-16 model's prediction
'''
## TODO: Complete the function.
## Load and pre-process an image from the given img_path
## Return the *index* oغf the predicted class for that image
img = Image.open(img_path)
normalize = transforms.Compose([
transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
img = normalize(img)
prediction = VGG16(img.unsqueeze(0))
prediction_int = int(torch.max(prediction.data, 1)[1].numpy())
return prediction_int # predicted class index
#Testing the method produces the following error:


print(VGG16_predict(human_files[20]))
