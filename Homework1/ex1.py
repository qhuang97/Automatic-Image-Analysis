#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Simon Matern
"""

import numpy as np
import cv2
import utils


def binarizeImage(img, thresh):
    """
    Given a coloured image and a threshold binarizes the image.
    Values below thresh are set to 0. All other values are set to 255
    """
    # TODO
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    th,img = cv2.threshold(gray,thresh,255,cv2.THRESH_BINARY)

    return img

def smoothImage(img):    
    """
    Given a coloured image apply a blur on the image, e.g. Gaussian blur
    """
    # TODO
    img = cv2.GaussianBlur(img,(13,13),0)
    return img

def doSomething(img):
    """
    Given a coloured image apply any image manipulation. Be creative!
    """
    # TODO
    blur = cv2.GaussianBlur(img,(3,3),0)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours,heriachy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,contours,-1,(0,255,255),2)

    return img


def processImage(img):
    """
    Given an coloured image applies the implemented smoothing and binarization.
    """
    # TODO
    img = smoothImage(img)
    img = binarizeImage(img, 125)
    return img


if __name__=="__main__":
    img = cv2.imread("test.jpg")
    utils.show(img)
    
    img1 = smoothImage(img)
    utils.show(img1)
    cv2.imwrite("result1.jpg", img1)
    
    img2 = binarizeImage(img, 125)
    utils.show(img2)
    cv2.imwrite("result2.jpg", img2)
   
    img3 = processImage(img)
    utils.show(img3)
    cv2.imwrite("result3.jpg", img3)
    
    img4 = doSomething(img)
    utils.show(img4)
    cv2.imwrite("result4.jpg", img4)
