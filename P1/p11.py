import cv2
img = cv2.imread('homework1/Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
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