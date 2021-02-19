
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
def smearDetection(image,path,imageNumber):

    # reading the input image
    image1 = cv2.imread(image)
    
    # convert image to gray scale
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # applying clahe filter to enhance the contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img1 = clahe.apply(img1)

    # applying gaussian and bilateral filter to reduce the noise
    img1 = cv2.GaussianBlur(img1, (7, 7), 0);
    img1 = cv2.bilateralFilter(img1, 9, 75, 75)

    # histogram of the image is generated and the max bin value is selected as thresh value
    hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
    thresh = np.argmax(hist)

    # thresholding to binarize the image
    ret, thresh1 = cv2.threshold(img1, thresh, 255, cv2.THRESH_BINARY)

    # erosion and dilation to enhance the smear contours
    kernel = np.ones((15, 15), np.uint8)
    erosion = cv2.erode(thresh1,kernel,iterations = 1)
    kernel = np.ones((10,10),np.uint8)
    thresh1 = cv2.dilate(erosion,kernel,iterations = 1)

    # finding the contours to approximate the smear
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filteredContours = []
    counter = 0
    max = 0
    # iteration over the contours to filter out contours based on the area
    # the limits for the area are decided heuristically
    for i in contours:
        area = cv2.contourArea(i)
        if (area > 10000 and area < 220000 ):
            filteredContours.append(i)
            counter = counter + 1
            if (area > max):
                max = area
    print('total number of contours: '+str(counter))

    # drawing the contours and generating the mask based on the image obtained after applying the contours
    contouredImage = cv2.drawContours(img1, filteredContours, -1, (255, 23, 71), 3)
    mask_inv = cv2.bitwise_not(contouredImage)
    res = cv2.bitwise_and(image1, image1, mask=mask_inv)
    numpy_horizontal_concat = np.concatenate((image1, res), axis=1)

    # adding the images to output directory if contours are found in the expected range
    if counter>0:
        cv2.imwrite(path+'result'+str(imageNumber)+'.jpg', numpy_horizontal_concat)
    
def main():
    smearDirectory = os.getcwd()+"/InputImages/*.jpg"
    outputDirectory = os.getcwd()+"/OutputImages/"
    list = glob.glob(smearDirectory)
    count =1
    for i in list:
        smearDetection(i,outputDirectory,count)
        count= count+1

if __name__ == "__main__":
    main()
