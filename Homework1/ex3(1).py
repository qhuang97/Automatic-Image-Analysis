#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:01:31 2021

@author: semjon
"""

import cv2
import utils
import numpy as np
from matplotlib import pyplot as plt

def nonMaxSuprression(img, d=5):
    """
    Given an image set all values to 0 that are not
    the maximum in this (2d+1,2d+1)-window
    给定图像将所有不等于0的值设置为0 （2d + 1,2d + 1）窗口中的最大值

    Parameters
    ----------
    img : ndarray
        an image
    d : int
        for each pixels consider the surrounding (2d+1,2d+1)-window

    Returns
    -------
    result : ndarray

    """
    rows,cols = img.shape
    result = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            low_y = max(0, i-d)
            low_x = max(0, j-d)
            
            high_y = min(rows, i+d) 
            high_x = min(cols, j+d) 
            
            max_val = img[low_y:high_y,low_x:high_x].max()
            
            if img[i,j] == max_val:
                result[i,j] = max_val
    print(f'max_val -> {max_val}')
    return result

def rotateAndScale(img, angle, scale):
    """
    Rotate and scale an image

    Parameters
    ----------
    img : ndarray
        an image
    angle : float
        angle given in degrees
    scale : float
        scaling of the image

    Returns
    -------
    result : ndarray
        a distorted image

    """
    
    h, w = img.shape
    (cX, cY) = (w // 2, h // 2) # // 向下取整

    M = cv2.getRotationMatrix2D((cX, cY), angle, scale)
    
    corners = np.array([[0, 0, 1],[0, h, 1], [w, 0, 1], [w, h, 1]]).T
    corners = M @ corners

    shift = corners.min(1)
    M[:,2]-= shift    
    
    b = corners.max(1)-corners.min(1)
    result = cv2.warpAffine(img, M, (int(b[0]),int(b[1])))
    return result

def calcDirectionalGrad(img):
    """
    Computes the gradients in x- and y-direction.
    The resulting gradients are stored as complex numbers.

    Parameters
    ----------
    img : ndarray
        an image

    Returns
    -------
    ndarray
        The array is stored in the following format: grad_x+ i*grad_y
    """
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    return sobelx + 1.0j*sobely


def circularShift(img, dx, dy):
    """
    Performs a circular shift and puts the new origin into position (dx,dy)
    进行循环移位并将新原点放置到位

    Parameters
    ----------
    img : ndarray
        an image
    dx : int
        x coordinate
    dy : int
        y coordinate

    Returns
    -------
    result : ndarray
        image with new center

    """
    img = img.copy()
    result = np.zeros_like(img)
    H,W = img.shape
    
    result[:-dy,:-dx] = img[dy:,dx:]
    result[:-dy,-dx:] = img[dy:,:dx]
    result[-dy:,:-dx] = img[:dy,dx:]
    result[-dy:,-dx:] = img[:dy,:dx]

    return result

def calcBinaryMask(img, thresh = 0.3):
    """
    Compute the gradient of an image and compute a binary mask
    based on the threshold. Corresponds to O^B in the slides.

    Parameters
    ----------
    img : ndarray
        an image
    thresh : float
        A threshold value. The default is 0.3.

    Returns
    -------
    binary : ndarray
        A binary image.

    """

    # TODO: 
    # -compute gradients
    # -threshold gradients 
    # -return binary mask
    
    oi = calcDirectionalGrad(img)
    ob = oi.copy()
    print("----------------")
    print(oi)
    ob[np.abs(ob) > thresh * np.max(np.abs(ob))] = 1
    ob[ob != 1] = 0
    
    return ob


def correlation(img, template):
    """
    Compute a correlation of gradients between an image and a template.
    
    Note:
    You should use the formula in the slides using the fourier transform.
    Then you are guaranteed to succeed.
    
    However, you can also compute the correlation directly. 
    The resulting image must have high positive values at positions
    with high correlation.

    Parameters
    ----------
    img : ndarray
        a grayscale image
    template : ndarray
        a grayscale image of the template

    Returns
    -------
    ndarray
        an image containing the correlation between image and template gradients.
    """
    
    # TODO:
    h, w = img.shape
    h_large, w_large = template.shape
    
    # 1-compute gradient of the image
    ii = calcDirectionalGrad(img)
    
    # 2-compute gradient of the template
    oi = calcDirectionalGrad(template)
    ob = calcBinaryMask(template, thresh = 0.3)
    T = oi * ob
    
    # 3-copy template gradient into larger frame
    
    lar_frame = np.zeros_like(img).astype(np.complex128)
    lar_frame[:T.shape[0],:T.shape[1]] += T 
    T_shift = circularShift(lar_frame, template.shape[1]//2, template.shape[0]//2)

    # 5-normalize template
    T_lar = T_shift / np.sum(abs(T_shift))
  
    # 6-compute correlation
    ii_fft = np.fft.fft2(ii)
    T_fft = np.fft.fft2(T_lar)
    
    cor = abs(np.fft.ifft2(ii_fft * T_fft))
    return cor
              
def GeneralizedHoughTransform(img, template, angles, scales):
    """
    Compute the generalized hough transform. Given an image and a template.
    
    Parameters
    ----------
    img : ndarray
        A query image
    template : ndarray
        a template image
    angles : list[float]
        A list of angles provided in degrees
    scales : list[float]
        A list of scaling factors

    Returns
    -------
    hough_table : list[(correlation, angle, scaling)]
        The resulting hough table is a list of tuples.
        Each tuple contains the correlation and the corresponding combination
        of angle and scaling factors of the template.
        
        Note the order of these values.
        hough_table：列表[（相关性，角度，缩放比例）]
        生成的hough表是一个元组列表。 每个元组都包含模板的相关性以及角度和比例因子的相应组合。
        注意这些值的顺序。
    """
    # TODO:
    # for every combination of angles and scales 
    # -distort template
    # -compute the correlation
    # -store results with parameters in a list
    list1 = []
    for i in angles:
        for s in scales:
            list0 = []
            disImage = rotateAndScale(template, i, s)
            cor = correlation(img, disImage)
            
            list0.append(cor)
            list0.append(i)
            list0.append(s)
            list1.append(tuple(list0)) 
            
    return list1 # 列表，360个数据，里面都为tuple

if __name__=="__main__":
    
    # Load query image and template 
    query = cv2.imread("data/query.jpg", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("data/template.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Visualize images
    utils.show(query)
    utils.show(template)

    # Create search space and compute GHT
    angles = np.linspace(0, 360, 36)
    scales = np.linspace(0.9, 1.3, 10)
    ght = GeneralizedHoughTransform(query, template, angles, scales)
    
    # extract votes (correlation) and parameters
    votes, thetas, s = zip(*ght)
    
    # Visualize votes
    votes = np.stack(votes).max(0)  
    plt.imshow(votes)
    plt.show()

    # nonMaxSuprression
    votes = nonMaxSuprression(votes, 20)
    plt.imshow(votes)
    plt.show()

    # Visualize n best matches
    n = 7
    coords = zip(*np.unravel_index(np.argpartition(votes, -n, axis=None)[-n:], votes.shape))
    vis = np.stack(3*[query],2)
    for y,x in coords:
        # print(x,y)
        vis = cv2.circle(vis,(x,y), 10, (255,0,0), 2)
    utils.show(vis)
