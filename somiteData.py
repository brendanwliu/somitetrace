import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import seaborn as sns
import cv2 as cv
import random
import re

from keras.preprocessing.image import ImageDataGenerator 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from PIL import Image

def center_crop(img, mask):
    imgCrop = img[(img.shape[0]//2 - 64):(img.shape[0]//2 + 64),(img.shape[1]//2 - 64):(img.shape[1]//2 + 64)]
    maskCrop = mask[(mask.shape[0]//2 - 64):(mask.shape[0]//2 + 64),(mask.shape[1]//2 - 64):(mask.shape[1]//2 + 64)]
    return (imgCrop, maskCrop)

def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode = "grayscale",
                    mask_color_mode = "grayscale", image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False, num_class = 2, save_to_dir = None, target_size = (128,128), seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    # we use the data generator for augmenting data - TODO later
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img, mask = center_crop(img, mask)
        yield (img, mask)

def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.tif"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)

#Add train, val, test frames and masks to relevant folders

def add_frames(dir_name, image):
    img = Image.open(FRAME_PATH+image)
    img.save(DATA_PATH+'/{}'.format(dir_name)+'/'+image)

def add_masks(dir_name, image):
    img = Image.open(MASK_PATH+image)
    img.save(DATA_PATH+'/{}'.format(dir_name)+'/'+image)

if __name__ == "__main__":
    DATA_PATH = 'datasets/SomiteTraceLibrary/input/'
    FRAME_PATH = DATA_PATH + 'frames/'
    MASK_PATH = DATA_PATH + 'masks/'

    # Create folders to hold images and masks

    folders = ['train_frames', 'train_masks', 'val_frames', 'val_masks', 'test_frames', 'test_masks']


    for folder in folders:
        os.makedirs(DATA_PATH + folder)
    
    
    # Get all frames and masks, sort them, shuffle them to generate data sets.

    all_frames = os.listdir(FRAME_PATH)
    all_masks = os.listdir(MASK_PATH)


    all_frames.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                    for x in re.findall(r'[^0-9]|[0-9]+', var)])
    all_masks.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])

    random.seed(230)
    temp = list(zip(all_frames, all_masks)) 
    random.shuffle(temp) 
    all_frames, all_masks = zip(*temp) 

    # Generate train, val, and test sets for frames

    train_split = int(0.7*len(all_frames))
    val_split = int(0.9 * len(all_frames))

    train_frames = all_frames[:train_split]
    val_frames = all_frames[train_split:val_split]
    test_frames = all_frames[val_split:]


    # Generate corresponding mask lists for masks

    train_masks = all_masks[:train_split]
    val_masks = all_masks[train_split:val_split]
    test_masks = all_masks[val_split:]

    frame_folders = [(train_frames, 'train_frames'), (val_frames, 'val_frames'), 
                    (test_frames, 'test_frames')]

    mask_folders = [(train_masks, 'train_masks'), (val_masks, 'val_masks'), 
                    (test_masks, 'test_masks')]

    # Add frames

    for folder in frame_folders:
    
        array = folder[0]
        name = [folder[1]] * len(array)
    
        list(map(add_frames, name, array))
            
        
    # Add masks

    for folder in mask_folders:
    
        array = folder[0]
        name = [folder[1]] * len(array)
    
        list(map(add_masks, name, array))