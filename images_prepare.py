# -*- coding: utf-8 -*-
"""

@author: serdarhelli
"""

import os
import numpy as np
from PIL import Image
import cv2
from zipfile import ZipFile
from natsort import natsorted


def convert_one_channel(img):
    # some images have 3 channels , although they are grayscale image
    if len(img.shape) > 2:
        img = img[:, :, 0]
        return img
    else:
        return img


def pre_images(resize_shape, path, include_zip):
    if include_zip == True:
        ZipFile(path+"/DentalPanoramicXrays.zip").extractall(path)
        path = path+'/Images/'
    dirs = natsorted(os.listdir(path))
    sizes = np.zeros([len(dirs), 2])
    images = img = Image.open(path+dirs[0])
    sizes[0, :] = images.size
    images = (images.resize((resize_shape), Image.ANTIALIAS))
    images = convert_one_channel(np.asarray(images))
    for i in range(1, len(dirs)):
        img = Image.open(path+dirs[i])
        sizes[i, :] = img.size
        img = img.resize((resize_shape), Image.ANTIALIAS)
        img = convert_one_channel(np.asarray(img))
        images = np.concatenate((images, img))
    images = np.reshape(
        images, (len(dirs), resize_shape[0], resize_shape[1], 1))
    return images, sizes, dirs


def pre_binary_images(resize_shape, path, include_zip):
    if include_zip == True:
        ZipFile(path+"/DentalPanoramicXrays.zip").extractall(path)
        path = path+'/Images/'
    dirs = natsorted(os.listdir(path))
    sizes = np.zeros([len(dirs), 2])
    images = img = Image.open(path+dirs[0])
    sizes[0, :] = images.size
    images = (images.resize((resize_shape), Image.ANTIALIAS))
    images = convert_one_channel(np.asarray(images))
    for i in range(1, len(dirs)):
        img = Image.open(path+dirs[i])
        sizes[i, :] = img.size
        img = img.resize((resize_shape), Image.ANTIALIAS)
        img = convert_one_channel(np.asarray(img))
        images = np.concatenate((images, img))
    images = np.reshape(
        images, (len(dirs), resize_shape[1], resize_shape[0], 1))
    return images, sizes, dirs


def pre_rgb_images(resize_shape, path, include_zip=False):
    import matplotlib.pyplot as plt
    if include_zip == True:
        ZipFile(path+"/DentalPanoramicXrays.zip").extractall(path)
        path = path+'/Images/'
    dirs = natsorted(os.listdir(path))
    sizes = np.zeros([len(dirs), 2])
    images = img = Image.open(path+dirs[0])
    sizes[0, :] = images.size
    images = (images.resize((resize_shape), Image.ANTIALIAS))
    # images=convert_one_channel(np.asarray(images))
    for i in range(1, len(dirs)):
        img = Image.open(path+dirs[i])
        sizes[i, :] = img.size
        img = img.resize((resize_shape), Image.ANTIALIAS)
        # img=convert_one_channel(np.asarray(img))
        images = np.concatenate((images, img))
    images = np.reshape(
        images, (len(dirs), resize_shape[0], resize_shape[1], 3))
    test = Image.fromarray(images[0, :, :, :])
    print(test.size)
    test.save('DataRGB/results/imgtestReshape.png')
    return images, sizes, dirs


def pre_rgb_images_nemo(resize_shape, path, include_zip=False):
    import matplotlib.pyplot as plt
    if include_zip == True:
        ZipFile(path+"/DentalPanoramicXrays.zip").extractall(path)
        path = path+'/Images/'
    dirs = natsorted(os.listdir(path))
    sizes = np.zeros([len(dirs), 2])
    images = img = Image.open(path+dirs[0])
    sizes[0, :] = images.size
    images = (images.resize((resize_shape), Image.ANTIALIAS))
    # images=convert_one_channel(np.asarray(images))
    for i in range(1, len(dirs)):
        img = Image.open(path+dirs[i])
        sizes[i, :] = img.size
        img = img.resize((resize_shape), Image.ANTIALIAS)
        # img=convert_one_channel(np.asarray(img))
        images = np.concatenate((images, img))
    images = np.reshape(
        images, (len(dirs), resize_shape[1], resize_shape[0], 3))
    return images, sizes, dirs


def pre_rgb_images(resize_shape, path):
    paths = natsorted(os.listdir(path))
    sizes = np.zeros([len(paths), 2])
    images = np.zeros((len(paths), resize_shape[0], resize_shape[1], 3))

    for i, imgname in enumerate(paths):
        img = cv2.imread(os.path.join(path, imgname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sizes[i, :] = [img.shape[0], img.shape[1]]
        img = cv2.resize(img, resize_shape)
        images[i, :, :, :] = img

    return images, sizes, paths
