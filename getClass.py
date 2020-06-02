#!/usr/local/bin/python2.7

import argparse as ap
import cv2
#import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *

# Load the classifier, class names, scaler, number of clusters and vocabulary 
clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

'''
# Get the path of the testing set
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument('-v',"--visualize", action='store_true')
args = vars(parser.parse_args())

def imlist(path):
    """
    The function imlist returns all the names of the files in 
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]


# Get the path of the testing image(s) and store them in a list
image_paths = []
if args["testingSet"]:
    test_path = args["testingSet"]
    try:
        testing_names = os.listdir(test_path)
    except OSError:
        print "No such directory {}\nCheck if the file exists".format(test_path)
        exit()
    for testing_name in testing_names:
        dir = os.path.join(test_path, testing_name)
        class_path = imlist(dir)
        image_paths+=class_path
else:
    image_paths = [args["image"]]
'''    
surf = cv2.xfeatures2d.SURF_create()

# List where all the descriptors are stored
def predict(gray):
    try:

        des_list = []
        (kps, des) = surf.detectAndCompute(gray, None)
        des_list.append(("image", des))   
    
# Stack all the descriptors vertically in a numpy array
        descriptors = des_list[0][1]
        for image_path, descriptor in des_list[0:]:
            descriptors = np.vstack((descriptors, descriptor)) 

# 
        test_features = np.zeros((1, k), "float32")
        for i in xrange(1):
            words, distance = vq(des_list[i][1],voc)
            for w in words:
                test_features[i][w] += 1

# Perform Tf-Idf vectorization
        nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
        idf = np.array(np.log((1.0*1+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scale the features
        test_features = stdSlr.transform(test_features)

# Perform the predictions
        predictions =  [classes_names[i] for i in clf.predict(test_features)]

        return predictions
    except:
        return []
