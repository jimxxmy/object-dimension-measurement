
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt 

def find_centroids(dst):
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 
                0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids[1:]),(5,5), 
              (-1,-1),criteria)
    return corners

image = cv2.imread("original.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

dst = cv2.cornerHarris(gray, 5, 3, 0.04)

dst = cv2.dilate(dst, None)


# Threshold for an optimal value, it may vary depending on the image.
# image[dst > 0.01*dst.max()] = [0, 0, 255]

# Get coordinates
corners= find_centroids(dst)
int_corners = np.asarray(corners,dtype=int)
# To draw the corners
l = []
l1 = []
for corner in int_corners:
    image[int(corner[1]), int(corner[0])] = [0, 0, 255]
  
    x = corner[0]
    y = corner[1]
    l.append(x)
    l1.append(y)
print ("The X Coordinates are: ",l)
print ("The Y Coordinates are: ",l1)
#Difference b/w X Coordinates
c_x = l[0]-l[1]
c1_x = l[1]-l[3]
c2_x = l[3]-l[4]
c3_x = l[4]-l[2]
c4_x = l[2]-l[0]
#difference b/w Y Coordinates
c_y = l1[0]-l1[1]
c1_y = l1[1]-l1[3]
c2_y = l1[3]-l1[4]
c3_y = l1[4]-l1[2]
c4_y = l1[2]-l1[0]
P1P2 = math.sqrt((pow(c_x,2)+pow(c_y,2)))
P2P3 = math.sqrt((pow(c1_x,2)+pow(c1_y,2)))
P3P4 = math.sqrt((pow(c2_x,2)+pow(c2_y,2)))
P4P5 = math.sqrt((pow(c3_x,2)+pow(c3_y,2)))
P5P0 = math.sqrt((pow(c4_x,2)+pow(c4_y,2)))


print('The pixels between point 1 and point 2 is:', P1P2)
print('The pixels between point 2 and point 3 is:', P2P3)
print('The pixels between point 3 and point 4 is:', P3P4)
print('The pixels between point 4 and point 5 is:', P4P5)
print('The pixels between point 5 and point 1 is:', P5P0)






#print(int_corners[1])    
cv2.imshow('dst', image)

cv2.imwrite('corners.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
