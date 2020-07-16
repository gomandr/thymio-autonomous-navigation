import cv2
import imageio
import math as m
import numpy as np
from PIL import Image
import scipy
import scipy.ndimage
from scipy import signal

def sobel_filter(im, k_size):
        im = im.astype(np.float)
        width, height, c = im.shape
        if c > 1:
            img = 0.2126 * im[:, :, 0] + 0.7152 * im[:, :, 1] + 0.0722 * im[:, :, 2]
        else:
            img = im
        assert (k_size == 3 or k_size == 5);
        if k_size == 3:
            kh = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float)
            kv = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float)
        else:
            kh = np.array([[-1, -2, 0, 2, 1],
                           [-4, -8, 0, 8, 4],
                           [-6, -12, 0, 12, 6],
                           [-4, -8, 0, 8, 4],
                           [-1, -2, 0, 2, 1]], dtype=np.float)
            kv = np.array([[1, 4, 6, 4, 1],
                           [2, 8, 12, 8, 2],
                           [0, 0, 0, 0, 0],
                           [-2, -8, -12, -8, -2],
                           [-1, -4, -6, -4, -1]], dtype=np.float)
        gx = signal.convolve2d(img, kh, mode='same', boundary='symm', fillvalue=0)
        gy = signal.convolve2d(img, kv, mode='same', boundary='symm', fillvalue=0)
        g = np.sqrt(gx * gx + gy * gy)
        g *= 255.0 / np.max(g)

        return g

def rectify_map(path):
    """
    :brief: rectify (pivot) and crop the original picture of the map
    :param path: string containing the path to the base picture of the map
    :return output_sobel: ndarray containing the edges of the input image (edges = '1', empty spaces = '0')
    :return rescaled: ndarray containing the rectified and cropped (rescaled) map contained in the input image
    :return rescaled_bw: ndarray containing the black and white version of rectified and cropped (rescaled)

    requires following packages:
        import cv2
        import imageio
        import math as m
        import numpy as np
        import os
        from PIL import Image
        import scipy
        import scipy.ndimage
        from scipy import signal

    references:
    [LI]
        LI, Feng, 2017. A simple implementation of sobel filtering in Python. Feng Li's Homepage [online]. 2017.02.28.
        Consulted on the 2019.12.07.
        Available at: https://fengl.org/2014/08/27/a-simple-implementation-of-sobel-filtering-in-python/

    [SO1]
        How to make a binary image with python?. Stack Overflow [online]. 2019.12.06.
        Consulted on 2019.12.07.
        Available at: https://stackoverflow.com/questions/43945749/how-to-make-a-binary-image-with-python

    [SO2]
        Python: exit out of two loops. Stack Overflow [online]. 2019.12.06.
        Consulted on 2019.12.07.
        Available at: https://stackoverflow.com/questions/3357255/python-exit-out-of-two-loops

    [SO3]
        How can I smooth elements of a two-dimensional array with differing gaussian functions in python?.
        Stack Overlfow [online]. 2019.12.06. Consulted on 2019.12.08.
        Available at: https://stackoverflow.com/questions/33548639/how-can-i-smooth-elements-of-a-two-dimensional-array-with-differing-gaussian-fun

    [OC]
        Geometric Transformations of Images. OpenCV [online]. 2019.12.07.
        Consulted on 2019.12.07.
        Available at: https://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
    """

    ## 1. Retrieving the 4 vertices
    # Sobel edge detection (in order to detect the four corner of the map)
    # Cf. [LI]
    im = imageio.imread(path)

    output_sobel = sobel_filter(im, 3)
    # Applying Gaussian filter to smooth output_sobel
    # Cf. [SO3]
    sigma_y = 2.0
    sigma_x = 2.0
    sigma = [sigma_y, sigma_x]
    output_sobel = scipy.ndimage.filters.gaussian_filter(output_sobel, sigma, mode='constant')
    THRESHOLD = 53 # THRESHOLD = 53 for Sobel (determined experimentally) with smoothing filter
    # Binarizing edge picture, cf. [SO], set anything less than THRESHOLD to 0
    output_sobel[output_sobel < THRESHOLD] = 0
    # Set all values greater thant THRESHOLD to 1
    output_sobel[output_sobel >= THRESHOLD] = 1
    gray = output_sobel
    colN = gray.shape[1]  # 3264 columns
    rowN = gray.shape[0]  # 2448 rows
    # ----------------------------
    #plt.figure();plt.imshow(gray)
    # ----------------------------
    # 1 = white, 0 = black
    index = np.where(gray == 1)
    corner_top = np.array([index[1][0], index[0][0]])
    corner_bottom = np.array([index[1][-1], index[0][-1]])
    # Retrieving corner_left (scrolling over the columns from the left)
    done = False
    for j in range(0, m.floor(colN / 2)):  # iterating over columns
        for i in range(0, rowN):           # iterating over lines
            if gray.item(i, j) == 1:
                corner_left = np.array([j, i])
                done = True
                break
        # Exiting two for loops, cf.: [SO2]
        if done:
            break
    # corner_left = array([ 384, 1924])
    # Retrieving corner_right (scrolling over the columns from the right)
    done = False
    for j in range(colN, m.floor(colN / 2), -1):  # iterating backwards over columns
        for i in range(0, rowN):  # iterating over rows
            if gray.item(i, j - 1) == 1:
                corner_right = np.array([j - 1, i])
                done = True
                break
        # Exiting two for loops, cf.: [SO2]
        if done:
            break

    ## 2. Rectifying the map
    # Cf. [OC]
    img = cv2.imread(path)
    # Actual Points
    pts1 = np.float32([corner_top, corner_right, corner_left, corner_bottom])
    # Target Points
    t1 = corner_top[0]
    t2 = corner_top[1]
    r1 = corner_right[0]
    r2 = corner_right[1]
    b1 = corner_bottom[0]
    b2 = corner_bottom[1]
    width = m.floor(m.sqrt((r1 - t1) ** 2 + (r2 - t2) ** 2))
    height = m.floor(m.sqrt((b1 - r1) ** 2 + (b2 - r2) ** 2))
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (width, height))

    ## 3. Rescaling image (between width and height)
    rescaled = dst[0:height, 0:width, :]

    ## 4. Converting "rescaled" to pure black and white
    # Pixels higher than THRESHOLD_VALUE will be 1, otherwise 0
    # 0 = black, 1 = white
    Image.fromarray(rescaled).save('rescaled.png')
    THRESHOLD_VALUE = 150
    # Convert image to greyscale
    img = Image.open('rescaled.png')
    #os.remove('rescaled.png')
    img = img.convert("L")
    imgData = np.asarray(img)
    gray2 = (imgData > THRESHOLD_VALUE) * 1.0
    # Gaussian blur
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    rescaled_bw = np.array(gray2)

    ## 5. Determining if rightTilted or leftTilted
    # (The program is constructed in such a way that rightTilted is considered by default)
    # Scrolling over the columns from left to right until reaching the first '1'
    done = False
    for j in range(0, m.floor(colN / 2)):  # iterating over columns
        for i in range(0, rowN):           # iterating over lines
            if gray.item(i, j) == 1:
                corner_detected = np.array([j, i])
                done = True
                break
        # Exiting two for loops, cf.: [SO2]
        if done:
            break
    # If the very first corner we detected (coming from the left and scrolling over the columns)
    # is situated below the upper half of the picture, it means we are in the default case: picture right tilted
    if (corner_detected[1] > rowN / 2):
        pass
    else:  # Else it means that the first detected corner is situated above the upper half of the picture,
           # which correspond to the case where the map taken in picture is left tilted.
           # Hence, we have to pivot the whole image of one quarter of turn
        rescaled_bw = np.rot90(rescaled_bw,k=3)
        rescaled = np.rot90(rescaled,k=3)

    return output_sobel, rescaled, rescaled_bw