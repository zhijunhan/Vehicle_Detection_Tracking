# Importing useful utilities and libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn import svm

from moviepy.editor import VideoFileClip

from utils import *

# Extract vehicle files and non-vehicle files from training data
# Read in vehicle and non-vehicle images
vehicle_files, vehicle_images = detector.extract_files_images('./data/vehicles/')
non_vehicle_files, non_vehicle_images = detector.extract_files_images('./data/non-vehicles/')

print('Number of vehicle files: {}'.format(len(vehicle_files)))
print('Number of non-vehicle files: {}'.format(len(non_vehicle_files)))

# Config the parameters for Support Vector Machine Classifier
clf_parameters = {'color_space':'YCrCb', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
				'orient':9, # HOG orientations
				'pix_per_cell':8, # HOG pixels per cell
				'cell_per_block':2, # HOG cells per block
				'hog_channel':'ALL', # Can be 0, 1, 2, or 'ALL'
				'spatial_size':(32,32), # Spatial binning dimensions
				'hist_bins':32, # Number of histogram bins
				'spatial_feat':True, # Spatial features on or off
				'hist_feat':True, # Histogram features on or off
				'hog_feat':True # HOG features on or off
				}

# Extract vehicle features
vehicle_features = detector.extract_features(vehicle_images, \
									clf_parameters['color_space'], \
									clf_parameters['orient'], \
									clf_parameters['spatial_size'], \
									clf_parameters['hist_bins'], \
									clf_parameters['pix_per_cell'], \
									clf_parameters['cell_per_block'], \
									clf_parameters['spatial_feat'], \
									clf_parameters['hist_feat'], \
									clf_parameters['hog_feat'], \
									clf_parameters['hog_channel'])

print('Size of the vehicle features: {}'.format(vehicle_features.shape))

# Extract non-vehicle features
non_vehicle_features = detector.extract_features(non_vehicle_images, \
										clf_parameters['color_space'], \
										clf_parameters['orient'], \
										clf_parameters['spatial_size'], \
										clf_parameters['hist_bins'], \
										clf_parameters['pix_per_cell'], \
										clf_parameters['cell_per_block'], \
										clf_parameters['spatial_feat'], \
										clf_parameters['hist_feat'], \
										clf_parameters['hog_feat'], \
										clf_parameters['hog_channel'])

print('Size of the non-vehicle features: {}'.format(non_vehicle_features.shape))

# Create an array stack of feature vectors
X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
# Create the labels vectors
y = np.hstack((np.ones(len(vehicle_images)), np.zeros(len(non_vehicle_images))))

# Cross validate classifier, should be enabled
#C, loss = detector.cross_validation(X, y)

# Normalize data features
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

# Create Linear Support Vector Classifier
#clf = LinearSVC(C=C, penalty='l2', loss=loss).fit(X, y)
clf = LinearSVC(C=0.08, penalty='l2', loss='hinge').fit(X, y)

# Create instance of vehicle detection and tracking
detector_ins = detector(clf_parameters=clf_parameters, scaler=scaler, classifier=clf)

# Test single image
#test_image = './test_images/test1.jpg'
#test_image = mpimg.imread(test_image)
#output_image = detector_ins.detect(test_image)
#plt.imshow(output_image)
#plt.show()

# Processing video frames
raw_clip = VideoFileClip('./test_video.mp4')
processed_clip = raw_clip.fl_image(detector_ins.detect)
processed_clip.write_videofile('./processed_project_video.mp4', audio=False)