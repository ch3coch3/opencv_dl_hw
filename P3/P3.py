import cv2
import numpy as np
import matplotlib.pyplot as plt

def conv2d(ori,kernal):
    ori = np.pad(ori,1,'constant',constant_values = 0)
    dimension = ori.shape
    result = np.zeros((250,461))
    for i in range(0,250):
        for j in range(0,461):
            ori_1 = ori[0+i:3+i,0+j:3+j]
            ori_2 = ori_1*kernal
            ori_2 = ori_2.sum()/kernal.sum()
            result[i,j] = int(ori_2)
    return result

def p1():
    img = cv2.imread("opencv_dl_hw\Dataset_opencvdl\Q3_Image\Chihiro.jpg")
    # filter
    sigma = 2
    x,y = np.mgrid[-1:2, -1:2]
    gaussian_kernal = np.exp(-(x**2 + y**2))/sigma**2
    gaussian_kernal = gaussian_kernal / gaussian_kernal.sum()

    # tranfer to gray scale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # convolution
    gaussian = conv2d(gray,gaussian_kernal)
    # plot figure
    plt.figure(1)
    plt.imshow(gaussian,cmap=plt.get_cmap('gray'))
    plt.title("original")
    plt.figure(2)
    plt.imshow(gray,cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.margins(0,0)
    plt.savefig("opencv_dl_hw/P3/Gaussian.png",bbox_inches = 'tight',dpi = 300,pad_inches = 0.0)
    plt.show()
