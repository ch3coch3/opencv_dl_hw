import cv2
def p1():
    img = cv2.imread('Q2_Image\Cat.png')
    median = cv2.medianBlur(img,7,dst=None)
    # cv2.imshow("img",img)
    cv2.imshow("median",median)



def p2():
    img = cv2.imread('Q2_Image\Cat.png')
    gaussian = cv2.GaussianBlur(img,(3,3),(3/2-1)*0.3+0.8)
    # cv2.imshow("img",img)
    cv2.imshow("gaussian",gaussian)



def p3():
    img = cv2.imread('Q2_Image\Cat.png')
    bil = cv2.bilateralFilter(img,9,90,90)
    # cv2.imshow("img",img)
    cv2.imshow("bilateral",bil)

