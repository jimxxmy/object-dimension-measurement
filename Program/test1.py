import cv2
import numpy as np
from matplotlib import pyplot as plt 

#Read Image of the Object
img = cv2.imread("C:\\Users\\Jimit\\Desktop\\Project\\captured.jpg")
cv2.imshow('Original Image', img)
cv2.waitKey(0)



#Convert Image To GrayScale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)
cv2.waitKey(0)


#Binary Thresholding
ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Binary Image', thresh)
cv2.waitKey(0)

#Crop Image
cropped = thresh[150:640, 150:500]
cv2.imshow('Cropped Image', cropped)
cv2.waitKey(0)





#Edge Detection
edges = cv2.Canny(cropped, 100, 200) 
cv2.imshow('Edges', edges)
cv2.waitKey(0)

#find contours
ctrs, hier = cv2.findContours(cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#Sort Contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * cropped.shape[1])


#ROI
for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = cropped[y:y + h, x:x + w]

    # show ROI
    # cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(cropped , (x, y), (x + w, y + h), (90, 0, 255), 2)

  

cv2.imshow('marked areas', cropped)
cv2.waitKey(0)
