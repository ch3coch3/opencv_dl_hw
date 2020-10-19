import cv2

img = cv2.imread("homework1\Dataset_opencvdl\Q1_Image\Flower.jpg")

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