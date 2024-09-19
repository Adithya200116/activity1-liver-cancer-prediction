#Before starting we should know what is countour exactly.
'''
Contours can be explained simply as a curve joining all the continuous points (along the boundary),
 having same color or intensity. 
 The contours are a useful tool for shape analysis and 
 object detection and recognition. 
 For better accuracy, use binary images.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# import the necessary packages
from skimage.feature import peak_local_max
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk
import cv2 as cv
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.layers import Input
from keras.models import Model
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
import keras
import pickle

global img

window = tk.Tk()

window.title("Liver cancer prediction")

window.geometry("700x710")
window.configure(background ="lightgray")

title = tk.Label(text="Click below to choose picture for testing liver cancer....", background = "lightgray", fg="Brown", font=("", 15))
title.grid()

def Random_Forest():
    df = pd.read_csv('dataset/indian liver patient.csv')
    df = df.reset_index()
    dist = df['Gender'].unique()
    distdict = {}
    num = 1
    for i in dist:
                    distdict[i] = num
                    num = num+1
    x = df[['Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio']]
    x = np.array(x)
    for i in x:
                    i[0] = int(distdict[i[0]])                                
    y = df['Result']
    y = np.array(y)
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    pickle.dump(clf,open('model.pkl','wb'))
    model=pickle.load(open('model.pkl','rb'))
    predict = clf.predict(X_test)
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    print("Accuracy : " + str(a) + "%")
    print("Precision : " +str(p)+"%")
    print("Recall : " +str(r)+"%")
    print("f1_score : " +str(f)+"%")

def analysis():
    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True,
    # 	help="path to input image")
    # args = vars(ap.parse_args())
    
    # # load the image and perform pyramid mean shift filtering
    # # to aid the thresholding step
    #image = cv2.imread(filename)
    global fileName
    # This is simply converting it to grayscale and applying the Otsu's thresholding.
    image = cv.imread(fileName,1)
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    cv2.imshow("Input", image)
    
    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
    	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)
    
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
    	labels=thresh)
    
    
    def generateModel():
      global classifier
      text.delete('1.0', END)
      if os.path.exists('model/model.json'):
          with open('model/model.json', "r") as json_file:
              loaded_model_json = json_file.read()
              classifier = model_from_json(loaded_model_json)
          classifier.load_weights("model/model_weights.h5")
          classifier._make_predict_function()   
          print(classifier.summary())
          f = open('model/history.pckl', 'rb')
          data = pickle.load(f)
          f.close()
          
    
      else:
          classifier = Sequential()
          classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 1), activation = 'relu'))
          classifier.add(MaxPooling2D(pool_size = (2, 2)))
          classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
          classifier.add(MaxPooling2D(pool_size = (2, 2)))
          classifier.add(Flatten())
          classifier.add(Dense(output_dim = 256, activation = 'relu'))
          classifier.add(Dense(output_dim = 1, activation = 'softmax'))
          print(classifier.summary())
          classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
          hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
          classifier.save_weights('model/model_weights.h5')            
          model_json = classifier.to_json()
          with open("model/model.json", "w") as json_file:
              json_file.write(model_json)
          f = open('model/history.pckl', 'wb')
          pickle.dump(hist.history, f)
          f.close()
          f = open('model/history.pckl', 'rb')
          data = pickle.load(f)
          f.close()
         
      
    
    
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm with active countors.
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    
    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
    	# if the label is zero, we are examining the 'background'
    	# so simply ignore it
    	if label == 0:
    		continue
    
    	# otherwise, allocate memory for the label region and draw
    	# it on the mask
    	mask = np.zeros(gray.shape, dtype="uint8")
    	mask[labels == label] = 255
    
    	# detect contours in the mask and grab the largest one
    	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    		cv2.CHAIN_APPROX_SIMPLE)
    	cnts = imutils.grab_contours(cnts)
    	c = max(cnts, key=cv2.contourArea)
    
    	# draw a circle enclosing the object
    	((x, y), r) = cv2.minEnclosingCircle(c)
    	cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
    	# cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
    	# 	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # show the output image
    cv2.imshow("Output", image)
    cv2.imwrite("../Result/Test/contour.jpg",image)                      #n    nbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb bn bn                                                                                          b
    cv2.waitKey(0)
    
def accuracy_finding():
    global fileName
    # This is simply converting it to grayscale and applying the Otsu's thresholding.
    img = cv.imread(fileName,1)
    #take image input and convert to grayscale
    #img = cv2.imread("F:\\Ideonix\\Python Projects\\liver -cancer-Image-Processing-Project (1)\\Dataset\\3.jpg",1);
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
    

def segmentation_func():
    global fileName
    # This is simply converting it to grayscale and applying the Otsu's thresholding.
    img = cv.imread(fileName,1)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    th1 = 0
    th2 = 255
    ret,thresh = cv.threshold(gray,th1,th2,cv.THRESH_BINARY + cv.THRESH_OTSU)
       
        # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
       
        # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
       
        # Defining accuracuy
    acc = 0.01
       
        # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,acc*dist_transform.max(),255,0)
       
        # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
       
        # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
     
        # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
       
        # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv.watershed(img,markers)
    img[markers == -1] = [0,0,255]
       
       
    cv.imwrite("../Result/Test/segmentation.jpg",img)
    cv.imshow("Result",img)
       
    cv.waitKey(0)
    cv.destroyAllWindows()


def openphoto():
    global fileName
    #dirPath = "./Dataset"
    #fileList = os.listdir(dirPath)
    #for fileName in fileList:
        #os.remove(dirPath + "/" + fileName)
    # you can change it according to the image location you have  
    fileName = askopenfilename(initialdir='./Dataset', title='Select image for analysis ',
                           filetypes=[('image files', '.jpg')])
    dst = "./Dataset"
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img1 = tk.Label(image=render, height="250", width="500")
    img1.image = render
    img = render
    img1.place(x=0, y=0)
    img1.grid(column=1, row=1, padx=10, pady = 10)
    title.destroy()
    #button1.destroy()
    #button2 = tk.Button(text="Analyse Image", command=analysis(fileName))
    #button2.grid(column=0, row=1, padx=10, pady = 10)
 
button0 = tk.Button(text="Upload Liver image", command = openphoto)
button0.grid(column=0, row=2, padx=10, pady = 10)

button1 = tk.Button(text="Predict Liver Cancer with Deep Learning Model", command = analysis)
button1.grid(column=0, row=6, padx=10, pady = 10)

button2 = tk.Button(text="Segmentation of Liver cancer image", command = segmentation_func)
button2.grid(column=0, row=10, padx=10, pady = 10)

button3 = tk.Button(text="Predict Cancer with VGG_16", command = accuracy_finding)
button3.grid(column=0, row=14, padx=10, pady = 10)

button3 = tk.Button(text="Predict Cancer with Random Forest", command = Random_Forest)
button3.grid(column=0, row=18, padx=10, pady = 10)

window.mainloop()



