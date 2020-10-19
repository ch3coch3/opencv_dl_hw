import cv2
img = cv2.imread('homework1/Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')

horizontal = cv2.flip(img,1)    # 橫向翻轉
# vertical = cv2.flip(img,0)      # 縱向翻轉
# together = cv2.flip(img,-1)     # 同時翻轉

cv2.imshow('My image',horizontal)
# cv2.imshow('My image',vertical)
# cv2.imshow('My image',together)
cv2.waitKey(0)
cv2.destroyAllWindows()