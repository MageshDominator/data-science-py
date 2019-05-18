# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:45:38 2019

@author: MAGESHWARAN
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def histvalue(image, channel="red"):
    """ Create Histogram values of an image

    Inputs:
        image (np.array) : Image input

        channel(string) : red (or) blue (or) green (or) all channel
                          on which histvalue to be calculated

    Output:
        hist (np.array) : (255, 1) array with hist values of a single channel
    """
    if channel=="red":
        hist = cv2.calcHist([image], channels=[2], mask=None, histSize=[256],
                            ranges=[0, 256])
        plt.plot(hist)

    elif channel=="green":
        hist = cv2.calcHist([image], channels=[1], mask=None, histSize=[256],
                            ranges=[0, 256])
        plt.plot(hist)

    elif channel=="blue":
        hist = cv2.calcHist([image], channels=[0], mask=None, histSize=[256],
                            ranges=[0, 256])
        plt.plot(hist)

    elif channel=="all":
        color = ("b", "g","r")

        for i, col in enumerate(color):
            hist = cv2.calcHist([image], channels=[i], mask=None, histSize=[256],
                            ranges=[0, 256])
            plt.plot(hist, color=col)
            plt.xlim(0, 256)
        plt.title("Histogram for all colors in the image")

    return hist

def equalizer(image):
    """ Histogram equalizer to increase the contrast

    Inputs:

        image (np.array) : Input Image

    Output:
        eq_image (np.array) : High contrast image
    """
    if len(image.shape) == 2:
        eq_image = cv2.equalizeHist(image)

    elif len(image.shape) == 3:
        # convert to hsv format
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # equalize the value channel of hsv image to adjust contrast
        hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])

        eq_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    return eq_image

if __name__=="__main__":
    # original image in opencv(bgr) format
    bgr_image = load_image("../image/data/logan.jpg", rgb=False)

    # rgb format
    rgb_image = load_image("../image/data/logan.jpg")

    hist = histvalue(bgr_image, channel="all")
    plt.savefig("../image/data/hist_logan.jpg")

    # gray image
    gray_image = load_image("../image/data/logan.jpg", gray=True)

    eq_image = equalizer(gray_image)

    # color image
    eq_image = equalizer(bgr_image)

