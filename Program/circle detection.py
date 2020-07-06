import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while(True):
    #Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow('LIVE FRAME!', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imwrite('C:\\Users\\Jimit\\Desktop\\Project\\Original Images\\Original Capture.jpg', frame)

img = cv2.imread('C:\\Users\\Jimit\\Desktop\\Project\\Original Images\\Original Capture.jpg',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,10,
                            param1=40,param2=30,minRadius=0,maxRadius=0)

print(circles)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
