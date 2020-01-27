import cv2
import numpy as np



img = cv2.imread("Clean_road.jpg", 0)
cv2.imwrite("canny.jpg", cv2.Canny(img, 200, 300))
cv2.imshow("canny", cv2.imread("canny.jpg"))
img = cv2.imread('canny.jpg')







noise_img =  cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)


#kernel = np.ones((3,3),np.uint8)
#dil = cv2.dilate(img,kernel)


#cv2.imshow('show',noise_img)

#cv2.waitKey()
#opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

#cv2.imshow('sho',opening)
#cv2.waitKey()

gray = cv2.cvtColor(noise_img, cv2.COLOR_BGR2GRAY) 
#gray = cv2.Canny(gray, 200, 300)
ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
_,contours,hierarchy=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hull = []
 
# calculate points for each contour
for i in range(len(contours)):
    # creating convex hull object for each contour
    hull.append(cv2.convexHull(contours[i], False))

drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
 
# draw contours and hull points
for i in range(len(contours)):
    color_contours = (0, 255, 0) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    # draw ith contour
    cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    cv2.drawContours(drawing, hull, i, color, 1, 8)
count =0
#print(cv2.contourArea(np.around(np.array([[pt] for pt in contours])).astype(np.int32)) )
for p in range(len(contours)):
    contour=np.array(contours[p]).astype(np.int32)
    #print(cv2.contourArea(contour))
    if cv2.contourArea(contour)>50:
        count+=1
#cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
print(' APPROXIMATE NO. OF POTHOLES = ' + str(count) )
cv2.imshow('img',drawing)
cv2.waitKey()

'''
###########333


th, im_th = cv2.threshold(opening, 220, 255, cv2.THRESH_BINARY_INV);
 
# Copy the thresholded image.
im_floodfill = im_th.copy()
 
# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
 
# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255);
 
# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
 
# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv
 
# Display images.
cv2.imshow("Thresholded Image", im_th)
cv2.imshow("Floodfilled Image", im_floodfill)
cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
cv2.imshow("Foreground", im_out)

'''

cv2.waitKey(0)














cv2.destroyAllWindows()


