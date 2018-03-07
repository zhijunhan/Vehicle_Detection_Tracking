import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os

from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define a main class object, which encapsulates methods created as above for
# feature extraction / generation, sliding windows, search through windows,
# and machine learning mode. Also, class object will remove duplicates and false positives.
# Plus, class object will perform the boxes drawing and retur nthe final processed image or video frame
class detector(object):
	def __init__(self, clf_parameters, scaler, classifier):
		# Vehicle detector attributes
		self.clf_parameters = clf_parameters
		# Search window specifications
		self.x_start_stop=[None, None]
		self.y_start_stop = [400,600]
		self.xy_window=(96, 85)
		self.xy_overlap=(0.75, 0.75)
		# Heat-map thresholding parameter
		self.heat_threshold = 2
		# Data feature normalization
		self.scaler = scaler
		# Class object classifier
		self.classifier = classifier
		# Video frame container
		self.frames = []

	# Define the spatial bining of color
	def bin_spatial(self, image, size=(32, 32)):
		# Use cv2.resize().ravel() to create the feature vector
		colorR = cv2.resize(image[:, :, 0], size).ravel()
		colorG = cv2.resize(image[:, :, 1], size).ravel()
		colorB = cv2.resize(image[:, :, 2], size).ravel()
		# Return the feature vector
		return np.hstack((colorR, colorG, colorB))	

	# Define a function to compute color histogram features
	def color_hist(self, img, nbins=32):
		# Compute the histogram of the RGB channels separately
		rhist = np.histogram(img[:, :, 0], bins=nbins)
		ghist = np.histogram(img[:, :, 1], bins=nbins)
		bhist = np.histogram(img[:, :, 2], bins=nbins)
		# Generating bin centers
		bin_edges = rhist[1]
		bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2
		# Concatenate the histograms into a single feature vector
		hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
		return hist_features

	@staticmethod
	# Classifier cross validation
	def cross_validation(X, y):
		# Define parameters for linear SVC
		param_C = [0.02, 0.08, 0.4, 1.0, 1.4]
		param_losses = ['hinge', 'squared_hinge']
		# Placeholders of classifier accuracy
		optimal_accu = 0.0
		# Iterate penalty, C and loss parameters to cross validate to find the best parameter
		for loss in param_losses:
			for C in param_C:
				# Shuffle train features and labels
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2048)
				# Fit a per-column scaler
				scaler = StandardScaler().fit(X_train)
				# Apply the scaler to train and test data features
				X_train = scaler.transform(X_train)
				X_test = scaler.transform(X_test)
				# Use a linear SVC
				svc = LinearSVC(C=C, penalty='l2', loss=loss).fit(X_train, y_train)
				# Calculate the classifier validation accuracy
				accuracy = svc.score(X_test, y_test)
				print('Validation accuracy: {:.4f} with C: {}, loss: {}'.format(accuracy, C, loss))
				# Cross check the accuracy and optimize
				if accuracy > optimal_accu:
						optimal_accu, optimal_C, optimal_loss = accuracy, C, loss
		print('The optimized validation accuracy is: {:.4f} with parameters: C: {}, loss: {}'.format(
				optimal_accu, optimal_C, optimal_loss))
		return optimal_C, optimal_loss

	# Define a function that takes an image, start and stop positions in both x and y, 
	# window size (x and y dimensions), and overlap fraction (for both x and y)
	@classmethod
	def slide_window(cls, \
					img, \
					x_start_stop=[None, None], \
					y_start_stop=[None, None], \
					xy_window=(64, 64), \
					xy_overlap=(0.5, 0.5)):
		# If x and/or y start/stop positions not defined, set to image size
		if x_start_stop[0] is None:
			x_start_stop[0] = 0
		if x_start_stop[1] is None:
			x_start_stop[1] = img.shape[1]
		if y_start_stop[0] is None:
			y_start_stop[0] = 0
		if y_start_stop[1] is None:
			y_start_stop[1] = img.shape[0]
		# Compute the span of the region to be searched
		xspan = x_start_stop[1] - x_start_stop[0]
		yspan = y_start_stop[1] - y_start_stop[0]
		# Compute the number of pixels per step in x/y
		nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
		ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
		# Compute the number of windows in x/y
		nx_windows = np.int(xspan / nx_pix_per_step) - 1
		ny_windows = np.int(yspan / ny_pix_per_step) - 1
		# Initialize a list to append window positions to
		window_list = []
		# Loop through finding x and y window positions
		# Note: you could vectorize this step, but in practice
		# you'll be considering windows one by one with your
		# classifier, so looping makes sense
		for ys in range(ny_windows):
			for xs in range(nx_windows):
				# Calculate window position
				startx = xs * nx_pix_per_step + x_start_stop[0]
				endx = startx + xy_window[0]
				starty = ys * ny_pix_per_step + y_start_stop[0]
				endy = starty + xy_window[1]
				# Append window position to list
				window_list.append(((startx, starty), (endx, endy)))
		# Return the list of windows
		return window_list	

	# Define a function you will pass an image
	# and the list of windows to be searched (output of slide_windows())
	@classmethod
	def search_windows(cls, \
						img, \
						windows, \
						clf, \
						scaler, \
						color_space, \
						spatial_size, \
						hist_bins, \
						orient, \
						pix_per_cell, \
						cell_per_block, \
						hog_channel, \
						spatial_feat, \
						hist_feat, \
						hog_feat):
		# 1) Create an empty list to receive positive detection windows
		on_windows = []
		# 2) Iterate over all windows in the list
		for window in windows:
			# 3) Extract the test window from original image
			test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
			# 4) Extract features for that window using extract_features()
			extracted_features = cls.extract_features([test_img], \
													color_space=color_space, \
													orient=orient, \
													spatial_size=spatial_size, \
													hist_bins=hist_bins, \
													pix_per_cell=pix_per_cell, \
													cell_per_block=cell_per_block, \
													spatial_feat=spatial_feat, \
													hist_feat=hog_feat, \
													hog_feat=hist_feat, \
													hog_channel=hog_channel)
			# 5) Scale extracted features to be fed to classifier
			test_features = scaler.transform(np.array(extracted_features).reshape(1, -1))
			# 6) Predict using classifier
			prediction = clf.predict(test_features)
			# 7) If possible (prediction == 1) then save the window
			if prediction == 1:
				on_windows.append(window)
		# 8) Return windows for positive detections
		return on_windows

	@staticmethod
	# Extract vehicle files and non-vehicle files from training data
	# Read in vehicle and non-vehicle images
	def extract_files_images(file_dir):
		files = []
		for dirpath, dirnames, filenames in os.walk(file_dir):
			for filename in filenames:
				if filename.endswith('.png'):
					files.append(os.path.join(dirpath, filename))
		images = [mpimg.imread(file) for file in files]
		return files, images

	@classmethod
	# Define a function to extract features from a list of images
	# Have this function call bin_spatial and color_hist
	def extract_features(cls, \
						images, \
						color_space, \
						orient, \
						spatial_size, \
						hist_bins, \
						pix_per_cell, \
						cell_per_block, \
						spatial_feat, \
						hist_feat, \
						hog_feat, \
						hog_channel):
		# Create a list to append feature vectors to
		features = []
		# Iterate through the list of images
		for image in images:
			file_features = []
			# Read in each one by one
			# image = mpimg.imread(image)
			# apply color conversion if other than 'RGB'
			if color_space != 'RGB':
				if color_space == 'HSV':
					feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
				elif color_space == 'LUV':
					feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
				elif color_space == 'HLS':
					feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
				elif color_space == 'YUV':
					feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
				elif color_space == 'YCrCb':
					feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
			else:
				feature_image = np.copy(image)
			if spatial_feat:
				spatial_features = cls.bin_spatial(cls, feature_image, size=spatial_size)
				# Append features to list
				file_features.append(spatial_features)
			# Compute histogram features if flag is set
			if hist_feat:
				hist_features = cls.color_hist(cls, feature_image, nbins=hist_bins)
				# Append features to list
				file_features.append(hist_features)
			# Call get_hog_features() with vis=False, feature_vec=True
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(cls.get_hog_features(
														feature_image[:, :, channel], \
														orient, \
														pix_per_cell, \
														cell_per_block, \
														vis=False, \
														feature_vec=True))
				hog_features = np.ravel(hog_features)
				file_features.append(hog_features)
			else:
				hog_features = cls.get_hog_features(
												feature_image[:, :, hog_channel], \
												orient, \
												pix_per_cell, \
												cell_per_block, \
												vis=False, \
												feature_vec=True)
				file_features.append(hog_features)
			# Append the new feature vector to the features list
			features.append(np.concatenate(file_features))
		# Return list of feature vectors
		return np.array(features)

	# Define a function to return HOG features and visualization
	@classmethod
	def get_hog_features(cls, \
						img, \
						orient, \
						pix_per_cell, \
						cell_per_block, \
						vis=False, \
						feature_vec=True):
		# Call with two outputs if vis == True
		if vis == True:
			features, hog_image = hog(img, \
										orientations=orient, \
										pixels_per_cell=(pix_per_cell, pix_per_cell), \
										cells_per_block=(cell_per_block, cell_per_block), \
										transform_sqrt=False, \
										visualise=True, \
										feature_vector=False)
			return features.ravel(), hog_image
		# Otherwise call with one output
		else:
			features = hog(img, \
							orientations=orient, \
							pixels_per_cell=(pix_per_cell, pix_per_cell), \
							cells_per_block=(cell_per_block, cell_per_block), \
							transform_sqrt=False, \
							visualise=False, \
							feature_vector=feature_vec)
			return features.ravel()

	# Class object memeber function
	# Apply thresholding on heatmap to filter out
	def apply_threshold(self, heatmap):
		# Zero out pixels below the threshold
		heatmap[heatmap <= self.heat_threshold] = 0
		# Return thresholded map
		return heatmap

	# Class object member function
	# Draw labeled bboxes on original images
	def draw_labeled_bboxes(self, img, labels):
		# Iterate through all detected cars
		for car_number in range(1, labels[1] + 1):
			# Find pixels with each car_number label value
			nonzero = (labels[0] == car_number).nonzero()
			# Identify x and y values of those pixels
			nonzeroy = np.array(nonzero[0])
			nonzerox = np.array(nonzero[1])
			# Define a bounding box based on min/max x and y
			bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
			# Draw the box on the image
			cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
		return img

	@classmethod
	def add_heat(cls, heatmap, bbox_list):
		# Iterate through list of bboxes
		for box in bbox_list:
			# Add += 1 for all pixels inside each bbox
			# Assuming each "box" takes the form ((x1, y1), (x2, y2))
			heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
		# Return updated heatmap
		return heatmap

	@staticmethod
	# Define a function to draw bounding boxes
	def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
		# Make a copy of the image
		imcopy = np.copy(img)
		# Iterate through the bounding boxes
		for bbox in bboxes:
			# Draw a rectangle given bbox coordinates
			cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
		# Return the image copy with boxes drawn
		return imcopy

	# Main class member function
	def detect(self, input_image):
		copy_image = np.copy(input_image)
		copy_image = copy_image.astype(np.float32) / 255.0

		slided_windows = self.slide_window(copy_image, \
										x_start_stop=self.x_start_stop, \
										y_start_stop=self.y_start_stop, \
										xy_window=self.xy_window, \
										xy_overlap=self.xy_overlap)
		on_windows = self.search_windows(copy_image, \
									slided_windows, \
									self.classifier, \
									self.scaler, \
									self.clf_parameters['color_space'], \
									self.clf_parameters['spatial_size'], \
									self.clf_parameters['hist_bins'], \
									self.clf_parameters['orient'], \
									self.clf_parameters['pix_per_cell'], \
									self.clf_parameters['cell_per_block'], \
									self.clf_parameters['hog_channel'], \
									self.clf_parameters['spatial_feat'], \
									self.clf_parameters['hist_feat'], \
									self.clf_parameters['hog_feat'])

		heat_map = np.zeros_like(copy_image)
		heat_map = self.add_heat(heat_map, on_windows)
		# Store frames
		self.frames.insert(0, heat_map)
		# Restrict the frames stored to be less than 25
		if len(self.frames) > 25:
			self.frames.pop()

		total_frames = np.array(self.frames)
		total_frames = np.sum(total_frames, axis=0)
		# Apply threshold to help remove false positives
		#heat_map = self.apply_threshold(total_frames)
		heat_map = self.apply_threshold(heat_map)
		# Figure out how many cars in each frame
		# Find final boxes from heatmap using label function
		labels = label(heat_map)
		# Overlay images with detected car boxes
		draw_img = self.draw_labeled_bboxes(input_image, labels)
		return draw_img
