import cv2
import numpy as np

img = cv2.imread('C:\\Users\\Jimit\\Desktop\\Project\\Original Images\\corners.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100, 250)

# get image height, width
(h, w) = img.shape[:2]
center = (w / 2, h / 2)


lines = cv2.HoughLinesP(edges,1,np.pi/90, 20, minLineLength = 50, maxLineGap = 200)

#print (lines)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print (line)


for theta in lines:
    a = np.cos(theta)
    b = np.sin(theta)

    a = np.rad2deg(a)
    b = np.rad2deg(b)

    print(a)
    print(b)



M = cv2.getRotationMatrix2D(center, 270, 1)

rotated = cv2.warpAffine(img, M, (w, h))
cv2.imshow('blah', rotated)
cv2.waitKey(0)
cv2.imshow('lmao', img)

cv2.imwrite('houghlines3.jpg',img)
