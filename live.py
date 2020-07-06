import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import os
import sys



def resource_path(relative_path):
 #Get absolute path to resource, works for dev and for PyInstaller
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

#Reference Object/Calibration
object_width = int(input("Enter the true width: "))
object_height = int(input("Enter the true height: "))
                                      
raw_image = cv2.imread('C:\\Users\\Jimit\\Desktop\\Project\\Ozeki_Captured_Raw_Image\\Original Capture.jpg')

#Read Image
image = cv2.imread('C:\\Users\\Jimit\\Desktop\\Project\\Ozeki_Captured_Raw_Image\\Original Capture.jpg')

#Find Corners
def find_centroids(dst):
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    #cv2.imshow('hm', dst)
    #cv2.waitKey(0)

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
dst = cv2.dilate(dst, None)

# Get coordinates of the corners.
corners = find_centroids(dst)

# To draw the corners
for i in range(0, len(corners)):
    print("Pixels found for this object are:",corners[i])
    image[dst>0.1*dst.max()] = [0,0,255]
    cv2.circle(image, (int(corners[i,0]), int(corners[i,1])), 7, (0,255,0), 2)

x_cord = []
y_cord = []
for corner in corners:
    image[int(corner[1]), int(corner[0])] = [0, 0, 255]
    
    x = corner[0]
    y = corner[1]
    x_cord.append(x)
    y_cord.append(y)



a = len(corners)
print("Number of corners found:",a)



#List to store pixel difference.
distance_pixel = []

#List to store mm distance.
distance_mm = []

#List to store Points Variable.
points_pixel = []

points_mm = []

#List to store Points name
points_name = []

if a == 4:
    print("Four corners found.")
    
    P1 = corners[0]
    P2 = corners[1]
    P3 = corners[2]
    P4 = corners[3]

    points_name.append("P1")
    points_name.append("P2")
    points_name.append("P3")
    points_name.append("P4")

    P1P2 = cv2.norm(P2-P1)
    P1P3 = cv2.norm(P3-P1)
    P2P4 = cv2.norm(P4-P2)
    P3P4 = cv2.norm(P4-P3)

    pixelsPerMetric_width1 = P1P2 / object_width
    pixelsPerMetric_width2 = P3P4 / object_width
    pixelsPerMetric_height1 = P1P3 / object_height
    pixelsPerMetric_height2 = P2P4 / object_height



    #Average of PixelsPerMetric
    pixelsPerMetric_avg = pixelsPerMetric_width1 + pixelsPerMetric_width2 + pixelsPerMetric_height1 + pixelsPerMetric_height2 

    pixelsPerMetric = pixelsPerMetric_avg / 4
    print(pixelsPerMetric)
    P1P2_mm = P1P2 / pixelsPerMetric
    P1P3_mm = P1P3 / pixelsPerMetric
    P2P4_mm = P2P4 / pixelsPerMetric
    P3P4_mm = P3P4 / pixelsPerMetric
    
    distance_mm.append(P1P2_mm)
    distance_mm.append(P1P3_mm)
    distance_mm.append(P2P4_mm)
    distance_mm.append(P3P4_mm)
    
    distance_pixel.append(P1P2)
    distance_pixel.append(P1P3)
    distance_pixel.append(P2P4)
    distance_pixel.append(P3P4)

    points_pixel.append("P1P2: ")
    points_pixel.append("P1P3: ")
    points_pixel.append("P2P4: ")
    points_pixel.append("P3P4: ")

    points_mm.append("P1P2: ")
    points_mm.append("P1P3: ")
    points_mm.append("P2P4: ")
    points_mm.append("P3P4: ")
    
if a == 5:
    print("Five corners found.")
    P1 = corners[0]
    P2 = corners[1]
    P3 = corners[2]
    P4 = corners[3]
    P5 = corners[4]

    points_name.append("P1")
    points_name.append("P2")
    points_name.append("P3")
    points_name.append("P4")
    points_name.append("P5")

    P1P2 = cv2.norm(P2-P1)
    P1P3 = cv2.norm(P3-P1)
    P2P4 = cv2.norm(P4-P2)
    P4P5 = cv2.norm(P5-P4)
    P3P5 = cv2.norm(P5-P3)

    
    
    distance_pixel.append(P1P2)
    distance_pixel.append(P1P3)
    distance_pixel.append(P2P4)
    distance_pixel.append(P4P5)
    distance_pixel.append(P3P5)

   

    pixelsPerMetric = P1P2 / object_width
    #print(pixelsPerMetric)
    P1P2_mm = P1P2 / pixelsPerMetric
    P1P3_mm = P1P3 / pixelsPerMetric
    P2P4_mm = P2P4 / pixelsPerMetric
    P4P5_mm = P4P5 / pixelsPerMetric
    P3P5_mm = P3P5 / pixelsPerMetric

    distance_mm.append(P1P2_mm)
    distance_mm.append(P1P3_mm)
    distance_mm.append(P2P4_mm)
    distance_mm.append(P4P5_mm)
    distance_mm.append(P3P5_mm)

    points_pixel.append("P1P2: ")
    points_pixel.append("P1P3: ")
    points_pixel.append("P2P4: ")
    points_pixel.append("P4P5: ")
    points_pixel.append("P3P5: ")

    points_mm.append("P1P2: ")
    points_mm.append("P1P3: ")
    points_mm.append("P2P4: ")
    points_mm.append("P4P5: ")
    points_mm.append("P3P5: ")
    



cv2.imshow('Corners Found!', image)
plt.imshow(image)
plt.show()

#Write text on image.
for x, y, b in zip(x_cord, y_cord, points_name):
    cv2.putText(raw_image, str(b), (x, y) , cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

#Write text on image.
for x, y, b in zip(x_cord, y_cord, points_name):
    cv2.putText(image, str(b), (x, y) , cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    
#Save Corner Image.
cv2.imwrite('C:\\Users\\Jimit\\Desktop\\Project\\Corners Found\\corners.jpg', image)
a = cv2.imread('C:\\Users\\Jimit\\Desktop\\Project\\Corners Found\\corners.jpg' )

#############################                  #########################################
#####GUI CODE#####

# GUI background image.
image_path = resource_path("white.jpg")

l1_img = cv2.imread(image_path)
## ORIGINAL CAPTURED IMAGE WINDOW
x_offset = 15
y_offset = 75
l1_img[y_offset:y_offset + raw_image.shape[0], x_offset:x_offset + raw_image.shape[1]] = raw_image 

x_offset = 15
y_offset = 75
l1_img[y_offset:y_offset + raw_image.shape[0], x_offset:x_offset + raw_image.shape[1]] = raw_image
# cv2.namedWindow('Output Window',cv2.WINDOW_NORMAL)

# TITLE TEXT
cv2.putText(l1_img, "*Computer Vision based Object Dimension Measurement**", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 0, 0), 1)

# PICTURE TITLE
cv2.putText(l1_img, "Original Captured Image", (150, 65), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

# PIXEL DISTANCE TEXT
cv2.putText(l1_img, "Distance(Pixels):", (15, 575), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)



#Points TEXT
offset = 20
x, y = 15, 590
for idx, points_pixel in enumerate(points_pixel):
    cv2.putText(l1_img, str(points_pixel), (x, y + offset * idx), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    
# offset for text
offset = 20
x, y = 70, 590
for idx, distance_pixel in enumerate(distance_pixel):
    cv2.putText(l1_img, str(distance_pixel), (x, y + offset * idx), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

## OUTPUT IMAGE WINDOW
x_offset = 665
y_offset = 75
l1_img[y_offset:y_offset + a.shape[0], x_offset:x_offset + a.shape[1]] = a

x_offset = 665
y_offset = 75
l1_img[y_offset:y_offset + a.shape[0], x_offset:x_offset + a.shape[1]] = a

# PICTURE TITLE
cv2.putText(l1_img, "Output Image", (900, 65), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

# MM DISTANCE TEXT
cv2.putText(l1_img, "Distance(MM):", (665, 575), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

#Points TEXT
offset = 20
x, y = 665, 590
for idx, points_mm in enumerate(points_mm):
    cv2.putText(l1_img, str(points_mm), (x, y + offset * idx), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

# offset for text
offset = 20
x, y = 720, 590
for idx, distance_mm in enumerate(distance_mm):
    cv2.putText(l1_img, str(distance_mm), (x, y + offset * idx), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)


cv2.imshow('Final Output Window',l1_img)
cv2.imwrite("C:\\Users\\Jimit\\Desktop\\Project\\Final Output Window\\Final Output Window.jpg", l1_img)
cv2.waitKey(0)


