"""
CS 6384 Homework 2 Programming
Implement the harris_corner() function and the non_maximum_suppression() function in this python script
Harris corner detector
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# TODO: implement this function
# input: R is a Harris corner score matrix with shape [height, width]
# output: mask with shape [height, width] with valuse 0 and 1, where 1s indicate corners of the input image
# idea: for each pixel, check its 8 neighborhoods in the image. If the pixel is the maximum compared to these
# 8 neighborhoods, mark it as a corner with value 1. Otherwise, mark it as non-corner with value 0
def non_maximum_suppression(R):
    mask = np.zeros_like(R)
    for y in range(R.shape[0]):
        for x in range(R[0].shape[0]):
            pixel_val = R[y, x]
            # top = max(0, y-1)
            # left = max(0, x-1)
            # right = min(R[0].shape[0]-1, x+1)
            # bottom = min(R.shape[0]-1, y+1)
            flag = -1
            for m in range(max(0, y-1), min(R.shape[0], y+2)):
                for n in range(max(0, x-1), min(R[0].shape[0], x+2)):
                    if y == m and x == n:
                        continue
                    # if R[m, n] >= pixel_val:
                    #     flag = 0
                    if flag < R[m, n]:
                        flag = R[m, n]

            if flag < pixel_val:
                # print(y, x)
                mask[y, x] = 1

    return mask


# TODO: implement this function
# input: im is an RGB image with shape [height, width, 3]
# output: corner_mask with shape [height, width] with valuse 0 and 1, where 1s indicate corners of the input image
# Follow the steps in Lecture 7 slides 29-30
# You can use opencv functions and numpy functions
def harris_corner(im):

    # step 0: convert RGB to gray-scale image
    # 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]
    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray, cmap=plt.get_cmap('gray'))
    # return gray

    # step 1: compute image gradient using Sobel filters
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # plt.subplot(2, 2, 1)
    # plt.imshow(gray, cmap='gray')
    # plt.imshow(sobelx, cmap='gray')
    # plt.imshow(sobely, cmap='gray')
    # plt.show()
    # return gray

    # step 2: compute products of derivatives at every pixels
    ix2 = np.multiply(ix, ix)
    iy2 = np.multiply(iy, iy)
    ixy = np.multiply(ix, iy)

    # step 3: compute the sums of products of derivatives at each pixel using Gaussian filter from OpenCV
    ix2 = cv2.GaussianBlur(src=ix2, ksize=(3, 3), sigmaX=3, sigmaY=0)
    iy2 = cv2.GaussianBlur(src=iy2, ksize=(3, 3), sigmaX=3, sigmaY=0)
    ixy = cv2.GaussianBlur(src=ixy, ksize=(3, 3), sigmaX=3, sigmaY=0)

    # step 4: compute determinant and trace of the M matrix
    detM = np.zeros_like(ix2)
    traceM = np.zeros_like(ix2)
    for y in range(ix2.shape[0]):
        for x in range(ix2[0].shape[0]):
            m = np.array([[ix2[y, x], ixy[y, x]], [ixy[y, x], iy2[y, x]]])
            detm = np.linalg.det(m)
            tracem = np.trace(m)
            detM[y, x] = detm
            traceM[y, x] = tracem
    # step 5: compute R scores with k = 0.05
    k = 0.05
    R = detM - k*(traceM*traceM)
    print(R.max())
    # step 6: thresholding
    # up to now, you shall get a R score matrix with shape [height, width]
    threshold = 0.01 * R.max()
    R[R < threshold] = 0

    # # step 7: non-maximum suppression
    # #TODO implement the non_maximum_suppression function above
    corner_mask = non_maximum_suppression(R)

    return corner_mask


# main function
if __name__ == '__main__':

    # read the image in data
    # rgb image
    rgb_filename = 'data/000006-color.jpg'
    im = cv2.imread(rgb_filename)

    # your implementation of the harris corner detector
    corner_mask = harris_corner(im)

    # opencv harris corner
    img = im.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    opencv_mask = dst > 0.01 * dst.max()

    # visualization for your debugging
    fig = plt.figure()

    # show RGB image
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(im[:, :, (2, 1, 0)])
    ax.set_title('RGB image')

    # show our corner image
    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(corner_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5)
    ax.set_title('our corner image')

    # show opencv corner image
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(opencv_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5)
    ax.set_title('opencv corner image')

    plt.show()
