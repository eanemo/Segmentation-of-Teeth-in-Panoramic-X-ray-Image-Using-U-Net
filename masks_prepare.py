# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from PIL import Image
import cv2
from zipfile import ZipFile
from natsort import natsorted

from tensorflow.keras.utils import to_categorical

script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
default_path = script_dir+'/Original_Masks/'


def convert_one_channel(img):
    # some images have 3 channels , although they are grayscale image
    if len(img.shape) > 2:
        img = img[:, :, 0]
        return img
    else:
        return img


def pre_masks(resize_shape=(512, 512), path=default_path):
    ZipFile(path+"/Orig_Masks.zip").extractall(path+'/Masks/')
    path = path+'/Masks/'
    dirs = natsorted(os.listdir(path))
    masks = img = Image.open(path+dirs[0])
    masks = (masks.resize((resize_shape), Image.ANTIALIAS))
    masks = convert_one_channel(np.asarray(masks))
    for i in range(1, len(dirs)):
        img = Image.open(path+dirs[i])
        img = img.resize((resize_shape), Image.ANTIALIAS)
        img = convert_one_channel(np.asarray(img))
        masks = np.concatenate((masks, img))
    masks = np.reshape(masks, (len(dirs), resize_shape[0], resize_shape[1], 1))
    return masks


default_path = script_dir+'/Custom_Masks/'


# CustomMasks 512x512
def pre_splitted_masks(path=default_path):
    ZipFile(path+"/splitted_masks.zip").extractall(path+'/Masks/')
    path = path+'/Masks/'
    dirs = natsorted(os.listdir(path))
    masks = img = Image.open(path+dirs[0])
    masks = convert_one_channel(np.asarray(masks))
    for i in range(1, len(dirs)):
        img = Image.open(path+dirs[i])
        img = convert_one_channel(np.asarray(img))
        masks = np.concatenate((masks, img))
    masks = np.reshape(masks, (len(dirs), 512, 512, 1))
    return masks

# CustomMasks_Nemo 512x512


def prepare_masks(path=default_path):
    path = path+'/Masks/'
    dirs = natsorted(os.listdir(path))
    masks = img = Image.open(path+dirs[0])
    masks = convert_one_channel(np.asarray(masks))
    for i in range(1, len(dirs)):
        img = Image.open(path+dirs[i])
        img = convert_one_channel(np.asarray(img))
        masks = np.concatenate((masks, img))
    masks = np.reshape(masks, (len(dirs), 512, 512, 1))
    return masks


def prepare_masks(resize_shape=(512, 512), path=default_path):
    dirs = natsorted(os.listdir(path))
    masks = img = Image.open(path+dirs[0])
    masks = (masks.resize((resize_shape), Image.ANTIALIAS))
    masks = convert_one_channel(np.asarray(masks))
    for i in range(1, len(dirs)):
        img = Image.open(path+dirs[i])
        img = img.resize((resize_shape), Image.ANTIALIAS)
        img = convert_one_channel(np.asarray(img))
        masks = np.concatenate((masks, img))
    masks = np.reshape(masks, (len(dirs), resize_shape[0], resize_shape[1], 1))
    return masks


def prepare_masks_nemo(resize_shape=(512, 512), path=default_path):
    dirs = natsorted(os.listdir(path))
    masks = img = Image.open(path+dirs[0])
    masks = (masks.resize((resize_shape), Image.ANTIALIAS))
    masks = convert_one_channel(np.asarray(masks))
    for i in range(1, len(dirs)):
        img = Image.open(path+dirs[i])
        img = img.resize((resize_shape), Image.ANTIALIAS)
        img = convert_one_channel(np.asarray(img))
        masks = np.concatenate((masks, img))

    masks = np.reshape(masks, (len(dirs), resize_shape[1], resize_shape[0], 1))
    return masks


def prepare_masks_onehot(path, nclasses, resize_shape=(512, 512), remove_background=False):
    dirs = natsorted(os.listdir(path))
    masks = np.zeros((len(dirs), resize_shape[0], resize_shape[1], nclasses))
    if remove_background:
        mask_classes = nclasses + 1
    else:
        mask_classes = nclasses

    for i, imgname in enumerate(dirs):
        img = Image.open(os.path.join(path, imgname))
        imgarray = np.asarray(img)
        # One-hot encoding multi-class segmentation
        msk_onehot = to_categorical(imgarray, mask_classes)
        msk = cv2.resize(msk_onehot, resize_shape)
        if remove_background:
            msk = msk[:, :, 1:nclasses+1]
        masks[i, :, :, :] = msk

    return masks


def prepare_color_masks(resize_shape=(512, 512), path=default_path):
    dirs = natsorted(os.listdir(path))
    masks = np.zeros((len(dirs), resize_shape[0], resize_shape[1], 3))

    for i, imgname in enumerate(dirs):
        img = Image.open(os.path.join(path, imgname))
        imgarray = np.asarray(img)
        msk = cv2.resize(imgarray, resize_shape)
        masks[i, :, :, :] = msk

    return masks


def prepare_masks_sparse(path, resize_shape=(512, 512)):
    dirs = natsorted(os.listdir(path))
    masks = np.zeros((len(dirs), resize_shape[0], resize_shape[1], 1))

    for i, imgname in enumerate(dirs):
        img = Image.open(os.path.join(path, imgname))
        imgarray = np.asarray(img)
        msk = cv2.resize(imgarray, resize_shape)
        masks[i, :, :, 0] = msk

    return np.array(masks, dtype='uint8')


"""
def prepare_masks_sparse(path, resize_shape=(512,512)):
    dirs=natsorted(os.listdir(path))
    masks = np.zeros((len(dirs), resize_shape[0] * resize_shape[1]))

    for i, imgname in enumerate(dirs):
        img = Image.open( os.path.join(path, imgname) )
        imgarray = np.asarray(img)
        msk = cv2.resize(imgarray, resize_shape)
        msk = msk.flatten()
        masks[i, :] = msk

    return np.array(masks, dtype='uint8')
"""
