import cv2

def p1():
    img = cv2.imread('opencv_dl_hw/Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
    dimension = img.shape

    height = dimension[0]
    width = dimension[1]
    channel = dimension[2]

    print("height=",height)
    print("width=",width)
    print(channel)

    cv2.imshow('My image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def p2():
    img = cv2.imread("opencv_dl_hw\Dataset_opencvdl\Q1_Image\Flower.jpg")

    # deal with red seperation(channel = 2)

    red_image = img.copy()
    red_image[:,:,0] = 0
    red_image[:,:,1] = 0
    cv2.imshow('red_image',red_image)

    # deal with red seperation(channel = 1)
    green_image = img.copy()
    green_image[:,:,0] = 0
    green_image[:,:,2] = 0
    cv2.imshow('green_image',green_image)

    # deal with red seperation(channel = 0)
    blue_image = img.copy()
    blue_image[:,:,1] = 0
    blue_image[:,:,2] = 0
    cv2.imshow('blue_image',blue_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def p3():
    img = cv2.imread('opencv_dl_hw/Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')

    horizontal = cv2.flip(img,1)    # 橫向翻轉
    # vertical = cv2.flip(img,0)      # 縱向翻轉
    # together = cv2.flip(img,-1)     # 同時翻轉

    cv2.imshow('My image',horizontal)
    # cv2.imshow('My image',vertical)
    # cv2.imshow('My image',together)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def p4():
    img = cv2.imread('opencv_dl_hw/Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
    horizontal = cv2.flip(img,1)    # 橫向翻轉\
    trans = 100/255
    def updateAlpha(x):
        global alpha, alpha2, blend
        alpha = (cv2.getTrackbarPos('BLEND','blend'))*0.01*trans
        alpha2 = (255 - cv2.getTrackbarPos('BLEND','blend')) *0.01*trans
        blend = cv2.addWeighted(img,alpha,horizontal,alpha2,0)
    cv2.namedWindow("blend")
    cv2.createTrackbar("BLEND","blend",0,255,updateAlpha)
    cv2.setTrackbarPos("BLEND","blend",128)
    while True:
        cv2.imshow("blend",blend)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()