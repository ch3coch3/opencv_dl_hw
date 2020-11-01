import cv2
import numpy as np

def open():
    img = cv2.imread("opencv_dl_hw\Dataset_opencvdl\Q4_Image\Parrot.png")
    cv2.imshow("original",img)


def p(angle,scale,tx,ty):
    angle = eval(angle)
    scale = eval(scale)
    tx = eval(tx)
    ty = eval(ty)
    img = cv2.imread("opencv_dl_hw\Dataset_opencvdl\Q4_Image\Parrot.png")
    row,col = img.shape[:2]

    # transformation
    trans = np.float32([[1,0,tx],[0,1,ty]])
    res = cv2.warpAffine(img,trans,(row,col))

    # Rotation
    ro = cv2.getRotationMatrix2D((160,84),angle,scale)
    res = cv2.warpAffine(res,ro,(row,col))
    cv2.imshow("rotation + trans + scale",res)

    cv2.waitKey(0)