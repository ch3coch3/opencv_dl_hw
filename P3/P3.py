import cv2
import numpy as np
import matplotlib.pyplot as plt

def conv2d(ori,kernal):
    dimension = ori.shape
    # ori = np.pad(ori,1,'constant',constant_values = 0)
    result = np.zeros((dimension[0],dimension[1]))
    for i in range(0,dimension[0]-2):
        for j in range(0,dimension[1]-2):
            ori_1 = ori[0+i:3+i,0+j:3+j]
            ori_2 = ori_1*kernal
            result[i,j] = np.abs(ori_2.sum())
    return result

def p1():
    img = cv2.imread("opencv_dl_hw\Dataset_opencvdl\Q3_Image\Chihiro.jpg")
    # filter
    sigma = 1
    x,y = np.mgrid[-1:2, -1:2]
    gaussian_kernal = np.exp(-(x**2 + y**2))/sigma**2
    gaussian_kernal = gaussian_kernal / gaussian_kernal.sum()
    # tranfer to gray scale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # convolution
    gaussian = conv2d(gray,gaussian_kernal)
    # transform to uint8
    gray = np.asarray(gray,dtype=np.uint8)
    gaussian = np.asarray(gaussian,dtype=np.uint8)
    
    # plot figure
    cv2.imwrite("opencv_dl_hw/P3/Gaussian.png",gaussian)
    cv2.imshow("gray",gray)
    cv2.imshow("gaussian",gaussian)


def p2():
    img = cv2.imread("opencv_dl_hw\P3\Gaussian.png")
    # img = np.asarray(img, dtype=np.int64)
    # filter
    sobelX_filter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    # tranfer to gray scale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # convolution
    sobelX = conv2d(gray,sobelX_filter)
    sobelX = np.asarray(sobelX,dtype = np.uint8)
    # plot figure
    cv2.imshow("SobelX",sobelX)

def p3():
    img = cv2.imread("opencv_dl_hw\P3\Gaussian.png")
    # filter
    sobelY_filter = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    # tranfer to gray scale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # convolution
    sobelY = conv2d(gray,sobelY_filter)
    sobelY = np.asarray(sobelY,dtype=np.uint8)
    # plot figure
    cv2.imshow("SobelY",sobelY)

def p4():
    img = cv2.imread("opencv_dl_hw\P3\Gaussian.png")
    # filter
    sobelX_filter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobelY_filter = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    # tranfer to gray scale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # convolution
    sobelX = conv2d(gray,sobelX_filter)
    sobelY = conv2d(gray,sobelY_filter)
    magnitude = np.asarray(np.sqrt(sobelX**2 + sobelY**2),np.uint8)
    # plot figure
    cv2.imshow("magnitude",magnitude)