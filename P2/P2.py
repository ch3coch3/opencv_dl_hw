import cv2
def p1():
    img = cv2.imread('opencv_dl_hw\Dataset_opencvdl\Q2_Image\Cat.png')
    median = cv2.medianBlur(img,7,dst=None)
    # cv2.imshow("img",img)
    cv2.imshow("median",median)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def p2():
    img = cv2.imread('opencv_dl_hw\Dataset_opencvdl\Q2_Image\Cat.png')
    gaussian = cv2.GaussianBlur(img,(3,3),(3/2-1)*0.3+0.8)
    # cv2.imshow("img",img)
    cv2.imshow("gaussian",gaussian)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def p3():
    img = cv2.imread('opencv_dl_hw\Dataset_opencvdl\Q2_Image\Cat.png')
    bil = cv2.bilateralFilter(img,9,90,90)
    # cv2.imshow("img",img)
    cv2.imshow("bilateral",bil)

    cv2.waitKey(0)
    cv2.destroyAllWindows()