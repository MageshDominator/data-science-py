# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:10:13 2019

@author: MAGESHWARAN
"""
import cv2
import matplotlib.pyplot as plt

def load_image(image_file, gray=False, rgb=True):

    """ Loading image from file system

    Inputs:
        image_file (string) : path of the image file

        gray (bool) : whether image to be loaded in grayscale

        rgb(bool) : opencv reads images in bgr, this heps us to convert image
                    to rgb
    Output:
        image (np.array) : array containing image
    """
    if gray:
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_file)
        if rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def display_image(image, gray=False):
    """ Displaying image

    Inputs:
        image (np.array) : array containing image

        gray (bool) : whether image is in grayscale
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    if gray:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)
