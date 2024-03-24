import os
import torch
import torchvision.transforms as transforms
from PIL import Image

# set the paths of the "depth" and "train" folders
train_folder = "/workspace/fog_3000/database/train/"
depth_folder = "/workspace/fog_3000/database/depth/"

# create a new folder named "concatenated"
concatenated_folder = "/workspace/fog_3000/database/input/"
if not os.path.exists(concatenated_folder):
    os.makedirs(concatenated_folder)

# define a transform to resize the images
transform = transforms.Resize((600, 400))

# get the filenames of all images in the "depth" folder
depth_filenames = os.listdir(depth_folder)

# loop through the filenames and concatenate the images
for filename in depth_filenames:
    # open the depth image and resize it
    depth_image = Image.open(os.path.join(depth_folder, filename))
    depth_image = transform(depth_image)
   
    # get the corresponding filename in the "train" folder
    train_filename = filename.replace("_depth", "")
   
    # open the train image and resize it
    train_image = Image.open(os.path.join(train_folder, train_filename))
    train_image = transform(train_image)
   
    # concatenate the images horizontally using torch.cat
    concatenated_image = torch.cat((transforms.ToTensor()(depth_image), transforms.ToTensor()(train_image)), dim=2)
   
    # convert the concatenated image back to a PIL Image
    concatenated_image = transforms.ToPILImage()(concatenated_image)
   
    # save the concatenated image to the "concatenated" folder
    concatenated_image.save(os.path.join(concatenated_folder, filename))