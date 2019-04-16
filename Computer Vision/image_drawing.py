# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:14:22 2019

Creator: Mageshwaran
"""
import cv2

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

# Reads image in BGR format
image = cv2.imread("./Logan.jpg", )

# -----------------------Image Color Processing--------------------------------
# MATPLOTLIB follows RGB channel so covert from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

# Read image in grayscale format
gray_image = cv2.imread("./Logan.jpg", cv2.IMREAD_GRAYSCALE)
plt.imshow(gray_image, cmap="gray")

# -----------------------------Resizing----------------------------------------
# Resize image directly with pixel values
resized_image = cv2.resize(image, (1800, 1200))

# Resize image with height and width ratio
w_ratio = 0.5 # 50 % of original width
h_ratio = 0.6 # 60% of original height
resized_image = cv2.resize(image, (0, 0), image, w_ratio, h_ratio)
plt.imshow(resized_image)


# drawing on images with values
image = cv2.imread("./Logan.jpg", )
cv2.rectangle(image, pt1=(1000, 200), pt2=(1500, 700), color=(0, 0, 255),
              thickness=20)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)


# Drawing on images with Mouse(Callback)
def draw_rectangle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(image, pt1=(x - 250 , y - 250), pt2=(x + 250, y + 250),
                      color=(0, 0, 255), thickness=20)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(image, center=(x, y), radius=250, color=(0, 255, 0),
                   thickness=20)

# Drawing on images with mouse drag
# InitiaLIZE GLOBAL VARIABLES
drawing =False
ix, iy = -1, -1

def drag_rectangle(event, x, y, flags, param):
    global drawing, ix, iy
    # if left button is clicked, set drawing as true
    # Initialize start point of rectangle(place where mouse is cicked) to ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    # Draw the recatngle over the moved axis using x, y position
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            pass
            # remove pass and uncomment nextline if the recatnge needs to be drwan at instant           
            # cv2.rectangle(image, pt1=(ix , iy), pt2=(x, y), color=(0, 0, 255), thickness=20)
    # when mouse button is set free, set drawing to False
    # draw the recatngle over that dragged axis
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(image, pt1=(ix , iy), pt2=(x, y), color=(0, 255, 0),
                      thickness=20)


cv2.namedWindow(winname="Detect_Logan")
# cv2.setMouseCallback("Detect_Logan", draw_rectangle)
cv2.setMouseCallback("Detect_Logan", drag_rectangle)

# ----------------------Displaying with CV window------------------------------
image = cv2.imread("./Logan.jpg", )

while True:
    cv2.imshow("Detect_Logan", image)
    # when esc key is pressed and we waited for 1ms then close the image window
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
