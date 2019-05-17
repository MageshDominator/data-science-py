# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:36:51 2019

@author: MAGESHWARAN
"""
import numpy as np
import cv2

def one_over_other(image_1, image_2, blend=False, resize=False, mask=False):

    """ Pasting images one over the other with or without blending

    Inputs:
        image_1 (np.array) : Larger image

        image_2 (np.array) : Smaller image

        blend (bool) : Blend images or not

        resize (bool) : resize or use region of interest

        maske (bool): adding masking layer
    Outputs:
        blended (np.array) : smaller image pasted over larger one with blending

        image_1 (np.array) : smaller image pasted over ROI of Larger image
    """

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
                # plt.imshow(fg)
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


def image_thresholding(image, high=255, low=127, adaptive=False,
                       threshold_type=cv2.THRESH_BINARY,
                       mean_type=cv2.ADAPTIVE_THRESH_MEAN_C, pix=11, noise=5):

    """ Coverting image to binary or some level of repr using thresholding

    Inputs:
        image (np.array) : Image to be modified
                            (for adaptive method must be in grayscale)

        high (int) : Maximum pixel value

        low (int) : threshold value pixels below which becomes 0

        adaptive (bool) : whether to do adaptive thresholding
                        (neighbour pixels are used)

        threshold_type (cv2 method) : comes with various options like Binary,
                            Binary inverse, Trunc

        mean_type (cv2 method) : If adaptive method is used,
                        this helps to select method for mean calculation

        pix (int) : Number of neighbour pixels to be looked at

        noise (int) : preferrably odd value, used to subtract mean
    Outputs:
        thresholded_image (np.array) : Thresholded image with adaptive / normal
                            thresholding
    """

    if not adaptive:
        rt, thresholded_image = cv2.threshold(image, low, high, threshold_type)

    else:
        thresholded_image = cv2.adaptiveThreshold(image, high, mean_type,
                                                 threshold_type, pix, noise)
    return thresholded_image

def blurring_smoothing(image, method, **kwargs):

    """ Blurring and Smoothing of images, Helps in removing noise from images

    Inputs:
        image (np.array) : input image

        method (str) : method to be applied on image -gamma_correction,
                    kernel, blur, gaussian_blur, median_blur

        gamma (float) : gamma value to adjust brightness
                        (for gamma correction method)

        kernel (np.array) : kernel filter used for image filtering

        kernel_size (tuple) : size of filter to be used when for blur method
                        Shape must be ODD for Gaussian_blur

        sigX (int) : Kernel standard deviation for X and Y

        kernel_shape (int) : size of kernel for median blur.
                        integer since it must be a square kernel(X==Y)

    Outputs:
        blurred (np.array) : Blurred / Smoothened image
    """

    if method == "gamma_correction":
        gamma = kwargs.pop("gamma")
        blurred = np.power(image, gamma)

    elif method == "kernel_filtering":
        kernel = kwargs.pop("kernel")
        blurred = cv2.filter2D(image, -1, kernel)

    elif method == "blur":
        kernel_size = kwargs.pop("kernel_size")
        blurred = cv2.blur(image, ksize=kernel_size)

    elif method == "gaussian_blur":
        kernel_size = kwargs.pop("kernel_size")
        sigX = kwargs.pop("sigX")
        blurred = cv2.GaussianBlur(image, kernel_size, sigX)

    elif method == "median_blur":
        kernel_shape = kwargs.pop("kernel_shape")
        blurred = cv2.medianBlur(image, kernel_shape)

    else:
        print("Enter proper method")

    return blurred


if __name__=="__main__":

    # reading images
    image = load_image("./data/joker.jpg") # Larger image
    image_ex = load_image("./data/example.jpg") # Smaller image

    # Blending and pasting of images
    new_image = one_over_other(image, image_ex, blend=True,
                           resize=False, mask=False)
    # display_image(new_image)

    img_grey = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
    # display_image(img_grey, True)

    # Thresholding----adaptive method works only on Grayscale image
    th_image = image_thresholding(img_grey, 255, 155, adaptive=True,
                                  threshold_type=cv2.THRESH_BINARY,
                                  mean_type=cv2.ADAPTIVE_THRESH_MEAN_C,
                                  pix=11, noise=8)

    to_blur_image = new_image.astype(np.float32) / 255
    # Blurring and smoothing of images

    # kernel matrix used for image filtering
    kernel = np.ones(shape=(5, 5), dtype=np.float32) / 80

    blur_image = blurring_smoothing(to_blur_image, "median_blur",
                                    kernel_shape=5)
    display_image(blur_image)