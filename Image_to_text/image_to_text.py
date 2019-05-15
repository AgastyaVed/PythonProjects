# Followed these instructions
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the file
img = cv2.imread('moneypak_card.jpg')

# display the file
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create a mask convert imgae to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)
def show_image(imageName,image):
    cv2.imshow(imageName,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Write the grayscale image to a file
cv2.imwrite('moneypak_card_gray.jpg',gray)

# Next step would be remove the background so that we can save a reduced image with just the receipt
mask = np.zeros(gray.shape[:2],np.uint8)

#thresholding the image: https://docs.opencv.org/3.1.0/d7/d4d/tutorial_py_thresholding.html
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2) # showing error. don't know why

# detecting edges
edges = cv2.Canny(gray,100,150 )

show_image('edgeImage',edges)
cv2.imwrite('edgeImage.jpg', edges)

#still not exactly what I want to do. I want to remove the empty space around the pic
# More exactly like this: https://docs.opencv.org/3.1.0/d8/d83/tutorial_py_grabcut.html
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (0,0,450,290)


cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img1 = img*mask2[:,:,np.newaxis]
show_image('img1', img1)

# The results don't look that good
# We need to know where is the ROI (Region of Interest) in our image
# https://docs.opencv.org/master/dc/df6/tutorial_py_histogram_backprojection.html
