# activity1-liver-cancer-prediction

# Project name: [Liver Cancer predication using ML]

# Author: [Adithya Kaushik]

This project focuses on image segmentation and analysis using various techniques and libraries, including OpenCV, NumPy, and scikit-learn. The primary goals are to perform effective image segmentation, thresholding, and classification, with applications in medical image analysis and other domains.

# Features

1. Image Segmentation
2. Thresholding
3. Blob Detection
4. Classification

# Dependencies used

## pip install numpy opencv-python scikit-learn matplotlib Pillow keras imutils

1. numpy
2. opencv-python
3. scikit-learn
4. matplotlib
5. Pillow
6. keras
7. imutils

# Usage

## Segmentation

Run the segmentation_func function to perform image segmentation. This function reads an image, applies Otsu’s thresholding, performs noise removal, and segments the image using the Watershed algorithm.

## Thresholding

The Hist function calculates the histogram of an image, and threshold function determines the optimal threshold using Otsu's method.

## Classification

Random Forest: Train and evaluate a Random Forest classifier on the liver patient dataset using Random_Forest.
Deep Learning: Use a pre-trained or newly trained Convolutional Neural Network (CNN) to predict liver cancer with the generateModel function.

## Blob Detection

The accuracy_finding function performs blob detection on images by applying Gaussian blur and detecting blobs using SimpleBlobDetector.

# Results

## Segmentation:

The segmented images are saved in the ../Result/Test/ directory.

## Thresholding:

Optimal thresholds and their effects on images are displayed and saved.

## Classification Metrics:

Accuracy, precision, recall, and F1 score for Random Forest and Deep Learning models are printed.

# Example output:

Accuracy : 64.95726495726495%
Precision : 54.642857142857146%
Recall : 51.92307692307692%
f1_score : 48.60173577627772%
