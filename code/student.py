import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature
from skimage.measure import regionprops

def plot_feature_points(image, x, y):
    '''
    Plot feature points for the input image. 
    
    Show the feature points given on the input image. Be sure to add the images you make to your writeup. 

    Useful functions: Some helpful (not necessarily required) functions may include
        - matplotlib.pyplot.imshow, matplotlib.pyplot.scatter, matplotlib.pyplot.show, matplotlib.pyplot.savefig
    
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of feature points
    :y: np array of y coordinates of feature points
    '''

    # TODO: Your implementation here! See block comments and the homework webpage for instructions
    plt.imshow(image)
    plt.scatter(x, y)

    plt.show()

def get_feature_points(image, window_width):
    '''
    Returns feature points for the input image.

    Implement the Harris corner detector.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.

    If you're finding spurious (false/fake) feature point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops
          
    Note: You may decide it is unnecessary to use feature_width in get_feature_points, or you may also decide to 
    use this parameter to exclude the points near image edges.

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :window_width: the width and height of each local window in pixels

    :returns:
    :xs: an np array of the x coordinates (column indices) of the feature points in the image
    :ys: an np array of the y coordinates (row indices) of the feature points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each feature point
    :scale: an np array indicating the scale of each feature point
    :orientation: an np array indicating the orientation of each feature point

    '''

    # These are placeholders - replace with the coordinates of your feature points!
    xs = np.random.randint(0, image.shape[1], size=100)
    ys = np.random.randint(0, image.shape[0], size=100)

    # STEP 1: Calculate the gradient (partial derivatives on two directions).
    # STEP 2: Apply Gaussian filter with appropriate sigma.
    # STEP 3: Calculate Harris cornerness score for all pixels.
    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)
    alpha = 0.06
    edges_y = filters.sobel_h(image)
    edges_x = filters.sobel_v(image)
    I_x_sqrt = edges_x * edges_x
    I_y_sqrt = edges_y * edges_y
    g_I_x_sqrt = filters.gaussian(I_x_sqrt, sigma=1)
    g_I_y_sqrt = filters.gaussian(I_y_sqrt, sigma=1)
    g_Ixy = filters.gaussian(I_x_sqrt * I_y_sqrt, sigma=1)
    cornerness = g_I_x_sqrt * g_I_y_sqrt - g_Ixy ** 2 - alpha * (g_I_x_sqrt + g_I_y_sqrt) ** 2
    local_max = feature.peak_local_max(cornerness, min_distance=15, threshold_abs=0, threshold_rel=0.005, exclude_border=10)
    xs = local_max[:, 1]
    ys = local_max[:, 0]


    return xs, ys


def get_feature_descriptors(image, x_array, y_array, window_width, mode):
    '''
    Returns features for a given set of feature points.

    To start with, use image patches as your local feature descriptor. You will 
    then need to implement the more effective SIFT-like feature descriptor. Use 
    the `mode` argument to toggle between the two.
    (Original SIFT publications at http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) feature descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4 x 4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    This is a design task, so many options might help but are not essential.
    - To perform interpolation such that each gradient
    measurement contributes to multiple orientation bins in multiple cells
    A single gradient measurement creates a weighted contribution to the 4 
    nearest cells and the 2 nearest orientation bins within each cell, for 
    8 total contributions.

    - To compute the gradient orientation at each pixel, we could use oriented 
    kernels (e.g. a kernel that responds to edges with a specific orientation). 
    All of your SIFT-like features could be constructed quickly in this way.

    - You could normalize -> threshold -> normalize again as detailed in the 
    SIFT paper. This might help for specular or outlier brightnesses.

    - You could raise each element of the final feature vector to some power 
    that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates (column indices) of feature points
    :y: np array of y coordinates (row indices) of feature points
    :window_width: in pixels, is the local window width. You can assume
                    that window_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like window will have an integer width and height).
    :mode: a string, either "patch" or "sift". Switches between image patch descriptors
           and SIFT descriptors

    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments. Make sure input arguments 
    are optional or the autograder will break.

    :returns:
    :features: np array of computed features. features[i] is the descriptor for 
               point (x[i], y[i]), so the shape of features should be 
               (len(x), feature dimensionality). For standard SIFT, `feature
               dimensionality` is typically 128. `num points` may be less than len(x) if
               some points are rejected, e.g., if out of bounds.
    '''

    # These are placeholders - replace with the coordinates of your feature points!
    features = np.random.randint(0, 255, size=(len(x_array), np.random.randint(1, 200)))

    # IMAGE PATCH STEPS
    # STEP 1: For each feature point, cut out a window_width x window_width patch 
    #         of the image (as you will in SIFT)
    # STEP 2: Flatten this image patch into a 1-dimensional vector (hint: np.flatten())
    
    # SIFT STEPS
    # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.
    # STEP 2: Decompose the gradient vectors to magnitude and orientation (angle).
    # STEP 3: For each feature point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the orientation (angle) of the gradient vectors. 
    # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
    #         we have a 128-dimensional feature.
    # STEP 5: Don't forget to normalize your feature.
    if mode == "patch":
        features = []
        for i, j in zip(y_array, x_array):
            patch = image[i - window_width//2 : i + window_width//2, j - window_width//2 : j + window_width//2]
            patch = np.asarray(patch).flatten()
            features.append(patch)
        features = np.asarray(features)

    else:
        grad_y = filters.sobel_h(image)
        grad_x = filters.sobel_v(image)
        grad_mag = np.sqrt(grad_x **2 + grad_y **2)
        grad_ori = np.arctan2(grad_y, grad_x)
        
        features = []
        for i, j in zip(y_array, x_array):
            descriptor = np.zeros((128, 1))
            w_grad_ori = grad_ori[i - window_width//2 : i + window_width//2, j - window_width//2 : j + window_width//2]
            w_grad_mag = grad_mag[i - window_width//2 : i + window_width//2, j - window_width//2 : j + window_width//2]
            for i in range(0, 16, 4):
                for j in range(0, 16, 4):
                    b_grad_ori = w_grad_ori[i:i+4, j:j+4]
                    b_grad_mag = w_grad_mag[i:i+4, j:j+4]
                    for b_i in range(4):
                        for b_j in range(4):
                            ori = b_grad_ori[b_i][b_j]
                            if -math.pi <= ori < (-3/4)*math.pi:
                                descriptor[i*2+0] += b_grad_mag[b_i][b_j]
                            elif (-3/4)*math.pi <= ori < (-1/2)*math.pi:
                                descriptor[i*2+1] += b_grad_mag[b_i][b_j]
                            elif (-1/2)*math.pi <= ori < (-1/4)*math.pi:
                                descriptor[i*2+2] += b_grad_mag[b_i][b_j]
                            elif (-1/4)*math.pi <= ori < 0:
                                descriptor[i*2+3] += b_grad_mag[b_i][b_j]
                            elif 0 <= ori < (1/4)*math.pi:
                                descriptor[i*2+4] += b_grad_mag[b_i][b_j]
                            elif (1/4)*math.pi <= ori < (1/2)*math.pi:
                                descriptor[i*2+5] += b_grad_mag[b_i][b_j]
                            elif (1/2)*math.pi <= ori < (3/4)*math.pi:
                                descriptor[i*2+6] += b_grad_mag[b_i][b_j]
                            else:
                                descriptor[i*2+7] += b_grad_mag[b_i][b_j]
            # n_descriptor = descriptor / np.linalg.norm(descriptor)
            features.append(descriptor)
        features = np.squeeze(np.asarray(features))

    return features


def match_features(im1_features, im2_features):
    '''
    Matches feature descriptors of one image with their nearest neighbor in the other. 
    Implements the Nearest Neighbor Distance Ratio (NNDR) Test to help threshold
    and remove false matches.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test".

    For extra credit you can implement spatial verification of matches.

    Remember that the NNDR will return a number close to 1 for feature 
    points with similar distances. Think about how you might want to threshold
    this ratio (hint: see lecture slides for NNDR)

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - np.argsort()

    :params:
    :im1_features: an np array of features returned from get_feature_descriptors() for feature points in image1
    :im2_features: an np array of features returned from get_feature_descriptors() for feature points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    '''

    # These are placeholders - replace with your matches and confidences!
    matches = np.random.randint(0, min(len(im1_features), len(im2_features)), size=(50, 2))

    A = np.sum(im1_features**2, axis=1).reshape(-1, 1) + np.sum(im2_features**2, axis=1).reshape(1, -1)
    B = 2 * (im1_features @ np.transpose(im2_features))
    distances = np.sqrt(A - B)

    sorted_distances = np.sort(distances)
    sorted_features = np.argsort(distances, axis=1)[:, 0]
    nearest_features = sorted_distances[:, 0]
    s_nearest_features = sorted_distances[:, 1]
    ratios = nearest_features / s_nearest_features

    thresholded_ratios_indices = np.where(ratios < 0.8)[0]
    matches = np.hstack((thresholded_ratios_indices.reshape(-1, 1), sorted_features[thresholded_ratios_indices].reshape(-1, 1)))

    
    # STEP 1: Calculate the distances between each pairs of features between im1_features and im2_features.
    #         HINT: https://browncsci1430.github.io/webpage/hw2_featurematching/efficient_sift/
    # STEP 2: Sort and find closest features for each feature
    # STEP 3: Compute NNDR for each match
    # STEP 4: Remove matches whose ratios do not meet a certain threshold
    

    return matches
