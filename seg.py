import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from glob import glob
import seaborn as sns



#########################################
BASE_DATA_FOLDER = "./data"
TRAin_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, "train")

#########################################
images_per_class = {}
for class_folder_name in os.listdir(TRAin_DATA_FOLDER):
    class_folder_path = os.path.join(TRAin_DATA_FOLDER, class_folder_name)
    class_label = class_folder_name
    images_per_class[class_label] = []
    for image_path in glob(os.path.join(class_folder_path, "*.png")):
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        images_per_class[class_label].append(image_bgr)


################ PRE-Processing  #########################
def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


#########################################


for key,value in images_per_class.items():
	c_name=key
	image = images_per_class[c_name][0]
	image_mask = create_mask_for_plant(image)
	image_segmented = segment_plant(image)
	image_sharpen = sharpen_image(image_segmented)

	fig, axs = plt.subplots(1, 4, figsize=(10, 5))
	axs[0].imshow(image)
	axs[1].imshow(image_mask)
	axs[2].imshow(image_segmented)
	axs[3].imshow(image_sharpen)

	plt.savefig('./segmented_image/Segmented-{0}.png'.format(key))

#########################################

