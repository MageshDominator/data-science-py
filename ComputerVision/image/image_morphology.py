# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:56:23 2019

@author: MAGESHWARAN
"""

import cv2
import numpy as np

def erode(image, kernel=np.ones((5, 5), dtype=np.uint8), iterations=1):
    """ Perform Erosion on image, this shrinks the image region based on kernel

    Inputs:
        image (np.array) : Image to be shrinked

        kernel (np.array): Filter to be used on the image

        iterations (int) : More the iteration, intense the erosion

    Output:
        result(np.array) : Eroded image
    """
    result = cv2.erode(image, kernel, iterations)
    return result


def dilate(image, kernel=np.ones((5, 5), dtype=np.uint8), iterations=1):
    """ Perform Dilation on image, this expands the image region based on kernel

    Inputs:
        image (np.array) : Image to be shrinked

        kernel (np.array): Filter to be used on the image

        iterations (int) : More the iteration, intense the erosion

    Output:
        result(np.array) : Dilated image
    """
    result = cv2.dilate(image, kernel, iterations)
    return result

def remove_noise(image, method="opening", kernel=np.ones((5, 5), dtype=np.uint8)):
    """ Removes noise form the image

    Inputs:
        image (np.array) : Image on which noise must be removed

        method(string)   : takes options opening(removes backgroiund noise),
                                         closing(removes foregroiund noise)

        kernel (np.array): Filter to be used on the image

    Output:
        result(np.array) : Noise free image
    """
    if method == "opening":
        image = image.astype(np.uint8)
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return result

    elif method == "closing":
        image = image.astype(np.uint8)
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return result

    else:
         print("Enter a valid method(opening/closing)")

if __name__ == "__main__":

    image = load_image("./data/binary.jpg", gray=True)

    # create a kernel to apply erosion and dilation
    kernel = np.ones((5, 5), dtype=np.uint8)

    # increase the iterations to see the shrinkage level
    eroded = erode(image, kernel=kernel, iterations=5)

    # increase the iterations to see the shrinkage level
    dilated = dilate(image, kernel=kernel, iterations=5)

    # -------------------- Remove BG noise with opening -----------------------
    white_noise = np.random.randint(low=0, high=2, size=((500, 500))) * 255

    bg_noisy_image = image + white_noise

    bg_noise_removed = remove_noise(bg_noisy_image, method="opening")

    # -------------------- Remove BG noise with opening -----------------------
    black_noise = np.random.randint(low=0, high=2, size=((500, 500))) * (-255)

    fg_noisy_image = image + black_noise

    fg_noisy_image[fg_noisy_image <= 0] = 0

    fg_noise_removed = remove_noise(fg_noisy_image, method="closing")

    display_image(fg_noisy_image, gray=True)
    cv2.imwrite("./data/fg_noise_removed.jpg", fg_noise_removed)
