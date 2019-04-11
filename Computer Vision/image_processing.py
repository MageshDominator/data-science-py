# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:36:51 2019

Creator: Mageshwaran
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("./joker.jpg") # Larger image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_ex = cv2.imread("./example.jpg") # smaller image
image_ex = cv2.cvtColor(image_ex, cv2.COLOR_BGR2RGB)

def one_over_other(image_1, image_2, blend=False, resize=False, mask=False):

    """ image_1 : Larger image
        image_2 : Smaller image
        blend   : Blend images or not
        resize  : resize or use region of interest
        maske   : adding masking layer"""

    # -----------------------overlay images with Blending----------------------
    # Resize the first image to second image size for blending
    image_w = image_2.shape[1]
    image_h = image_2.shape[0]
    if blend:
        if image_1.shape == image_2.shape:
            blended = cv2.addWeighted(src1=image_1, alpha=0.5,
                                      src2=image_2, beta=0.3, gamma=0.1)
            # it works only for images with same size

        elif resize:
            image_1 = cv2.resize(image_1, (image_w, image_h))
            # pasting rezied image over the first image
            blended = cv2.addWeighted(src1=image_1, alpha=0.5,
                                      src2=image_2, beta=0.3, gamma=0.1)

        else:
            # create region of interest in larger image and overlay over it
            # without masking the smaller image
            x_offset = image_1.shape[1] - image_w
            y_offset = image_1.shape[0] - image_h
            roi = image_1[y_offset:image_1.shape[0], x_offset:image_1.shape[1]]

            if not mask:
                blended = cv2.addWeighted(src1=roi, alpha=0.5,
                                          src2=image_2, beta=0.3, gamma=0.1)

            else:
                # convert image to grayscale
                # find inverse of that, which can be masked
                image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)
                image_2_inv = cv2.bitwise_not(image_2_gray)
                # create a image with fully white background :: 255
                # full_white = np.full(image_2_inv.shape, 255, dtype=np.uint8)
                # perform bitwise or with image_2 and image_2_inv
                # bk = cv2.bitwise_or(full_white, full_white, mask=image_2_inv)
                fg = cv2.bitwise_or(image_2, image_2, mask=image_2_inv)
                blended = cv2.bitwise_or(roi, fg)
                plt.imshow(fg)
        return blended
    else:
        # ---------------------Overlay images without Blending-----------------
        # slicing of larger image based on smaller image dimensions
        # then pasting smaller image over it
        x_offset = 0
        y_offset = 0
        x_end = x_offset + image_w
        y_end = y_offset + image_h
        image_1[y_offset:y_end, x_offset:x_end] = image_2
        return image_1

new_image = one_over_other(image, image_ex, blend=True,
                           resize=False, mask=False)

# Coverting image to binary or some level of repr using thresholding
def image_thresholding(image, high=255, low=127, adaptive=False,
                       threshold_type=cv2.THRESH_BINARY,
                       mean_type=cv2.ADAPTIVE_THRESH_MEAN_C, pix=11, noise=5):

    """ image          : Image to be modified(must be in grayscale)
        high           : Maximum pixel value
        low            : threshold value pixels below which becomes 0
        adaptive       : whether to do adaptive thresholding
                        (neighbour pixels are used)
        threshold_type : comes with various options like Binary,
                            Binary inverse, Trunc
        mean_type      : If adaptive method is used, this helps to select
                        method for mean calculation
        pix            : Number of neighbour pixels to be looked at
        noise          : constant used to subtract mean"""

    if not adaptive:
        rt, thresholded_image = cv2.threshold(image, low, high, threshold_type)

    else:
        thresholded_image = cv2.adaptiveThreshold(image, high, mean_type,
                                                 threshold_type, pix, noise)
    return thresholded_image

img_grey = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
plt.imshow(img_grey, cmap="gray")

# works only on Grayscale image
th_image = image_thresholding(img_grey, 255, 155, adaptive=True,
                              threshold_type=cv2.THRESH_BINARY,
                              mean_type=cv2.ADAPTIVE_THRESH_MEAN_C,
                              pix=11, noise=8)

plt.imshow(th_image, cmap="gray")
