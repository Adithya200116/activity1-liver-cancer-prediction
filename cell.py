import cv2
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk

def accuracy_finding():
    #take image input and convert to grayscale
    img = cv2.imread("F:\\Ideonix\\Python Projects\\liver -cancer-Image-Processing-Project (1)\\Dataset\\3.jpg",1);
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # take the gaussian blur image
    gauss = cv2.GaussianBlur(img,(5,5),100)
    
    #find threshold to convert into pure black white image
    ret,thresh = cv2.threshold(gauss,127,255,0)
    
    #detect holes using blob detector
    detector = cv2.SimpleBlobDetector_create()
    keypoint = detector.detect(thresh);
    imgkeypoint = cv2.drawKeypoints(thresh,keypoint,np.array([]),(0,255,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    #show the image on to the screen
    cv2.imshow("liver",imgkeypoint);
    cv2.waitKey(0)
    cv2.destroyAllWindows()
