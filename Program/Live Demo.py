
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
while(True):
    #Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow('Original Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imwrite('C:\\Users\\Jimit\\Desktop\\Project\\Original Images\\Original Capture.jpg', frame)


#Read Image
image = cv2.imread('C:\\Users\\Jimit\\Desktop\\Project\\Original Images\\Original Capture.jpg')

#Find Corners
def find_centroids(dst):
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    cv2.imshow('hm', dst)
    cv2.waitKey(0)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 
                0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids[1:]),(5,5), 
              (-1,-1),criteria)
    return corners



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

dst = cv2.cornerHarris(gray, 5, 3, 0.04)

#dst = cv2.dilate(dst, None)




# Get coordinates of the corners.
corners= find_centroids(dst)

# To draw the corners


for i in range(1, len(corners)):
    print(corners[i])
    image[dst>0.1*dst.max()] = [0,0,255]
    cv2.circle(image, (int(corners[i,0]), int(corners[i,1])), 7, (0,255,0), 2)
    
cv2.imshow('Corners Found!', image)
plt.imshow(image)
plt.show()
cv2.imwrite('C:\\Users\\Jimit\\Desktop\\Project\\Corners Found\\corners.jpg', image)
cv2.waitKey(0)
               
               

               








