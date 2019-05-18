# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:31:06 2019

@author: MAGESHWARAN
"""
import cv2
import numpy as np
from image_processing import one_over_other

def detect_edges(image, kernel=np.ones((5, 5), dtype=np.uint8)):
    """ Perform Edge detection on the image using Morphology Gradient

    Inputs:
        image (np.array) : input Image

        kernel (np.array): Filter to be used on the image

    Output:
        result(np.array) : Image with Edges detected
    """
    image = image.astype(np.uint8)
    result = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return result

def gradients(image, method="laplacian", ksize=5, **kwargs):
    """ Perform Edge detection on the image using sobel or laplace methods

    Inputs:
        image (np.array) : input Image

        method (string)  : either sobel or laplacian

        ksize (int) : Size of the kernel to be used

        axis (int)       : 0 for sobel operation in 'x' axis
                           1 for sobel operation in 'y' axis
                           2 for sobel operation in 'x,y' axis


    Output:
        result(np.array) : Image with Edges detected
    """
    if method == "sobel":
        axis = kwargs.pop("axis")
        if axis == 0:
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
            return sobel_x
        elif axis == 1:
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
            return sobel_y
        else:
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
            sobel = one_over_other(sobel_x, sobel_y, blend=True)
            return sobel

    elif method == "laplacian":
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
        return laplacian

if __name__=="__main__":

    # ------------------------- Edge Detection --------------------------------
    image = load_image("./data/binary.jpg", gray=True)

    edge_detected = detect_edges(image)

    # Sobel operator for edge detection
    sudoku = load_image("./data/sudoku.jpg", gray=True)

    sobel = gradients(sudoku, method="sobel", axis=2, ksize=5)

    display_image(sobel, gray=True)

    cv2.imwrite("./data/sobel_xy.jpg", sobel)