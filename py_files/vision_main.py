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

class MapNode:
    def __init__(self, x, y, is_obstacle=False, ground='unknown'):
        """
        :param x: int
        :param y: int
        :param is_obstacle: bool, optional
        """
        self.x = x
        self.y = y
        self.is_obstacle = is_obstacle
        self.ground = ground
    def setGround(self, ground):
        """
        :brief: Set the color of the node ('white', 'black' or 'unknown')
        :param ground: 'white', 'black' or 'unknown'
        """
        self.ground = ground
    def getGround(self):
        """
        :brief: Check the color of the ground
        :return: 'white', 'black' or 'unknown'
        """
        return self.ground
    def setObstacle(self, is_obstacle):
        """
        :brief: Set the node to obstacle (True or False)
        :param is_obstacle: bool
        """
        self.is_obstacle = is_obstacle
    def isObstacle(self):
        """
        :brief: Check if the node is an obstacle
        :return: bool
        """
        return self.is_obstacle
    def __repr__(self):
        """
        :brief: This is called when the class is printed as is
        :return: str
        """
        if self.is_obstacle:
            return "x"
        else:
            return " "

def remapping(rescaled_bw, mask_obst, mask_Thymio, dimX, dimY, start_pxl):
    """
    :brief: discretizes the input picture by identifying the the robot pose and the fixed obstacles
    :param rescaled_bw: ndarray containing the black and white version of rectified and cropped (rescaled)
    :param mask_obst: ndarray containing the points of location of the obstacles
    :param mask_Thymio: ndarray containing the points of location of the robot
    :param dimX: number of boxes on the width of our rescaled picture
    :param dimY: number of boxes on the height of our rescaled picture
    :param start_pxl: list containing the pose of the robot in [pixel] and its orientation in degrees
    :return Map: ndarray of class MapNode containing information about the color of the ground and
            the presence of obstacle for every coordinate
    :return Pattern: ndarray representing the pattern on the ground (0 = white, 1 = black, 2 = unknown)
    :return Obstacles: ndarray representing the obstacles (0 = free areas, 1 = obstacles)
    :return start: list containg the pose of the robot according to the coordinate of the discretized map and
            its orientation in degrees

    requires following packages:
        import math as m
        import numpy as np
    """

    # Setting dimensions of the map that the robot will explore (in [pxl])
    dimX_pxl = mask_obst.shape[1]
    dimY_pxl = mask_obst.shape[0]
    # Create a map and attribute coordinates to each of the map nodes using a reshaped list
    Map = []
    for row in range(dimY):
        for col in range(dimX):
            Map.append(MapNode(x=row, y=col, is_obstacle=False, ground='unknown'))
    Map = np.array(Map)
    Map = Map.reshape(dimY,dimX)

    # Edge of a black or white box
    edgeX = m.floor(dimX_pxl/dimX)
    edgeY = m.floor(dimY_pxl/dimY)

    for j in range(0,dimY):
        for i in range(0,dimX):

            ## Borders of the map
            # Setting the border of the map as obstacle
            if ( j==0 or i==0 or j==(dimY-1) or i==(dimX-1) ):
                Map[j][i].setObstacle('True')

            # Moving window on mask_obst
            window_mask = mask_obst[(j * edgeY):((j + 1) * edgeY), (i * edgeX):((i + 1) * edgeX)]

            # Moving window on mask_Thymio
            window_Thymio = mask_Thymio[(j * edgeY):((j + 1) * edgeY), (i * edgeX):((i + 1) * edgeX)]

            ## Fixed obstacles
            # Taking mean in order to discard unwanted spots
            if window_mask.mean()>127:
                Map[j][i].setObstacle('True')
                # Increasing the size of the real obstacles by adding a few .setObstacle boxes
                # so that we take the width of the robot into account (and that none of its parts
                # go over an obstacle)
                #----------------------------------
                # Big cross
                # Big cross - y
                if j > 0:
                    Map[j - 1][i].setObstacle('True')
                if j-1 > 0:
                    Map[j - 2][i].setObstacle('True')
                if j < (dimY - 1):
                    Map[j + 1][i].setObstacle('True')
                if j+1 < (dimY - 1):
                    Map[j + 2][i].setObstacle('True')
                # Big cross - x
                if i > 0:
                    Map[j][i - 1].setObstacle('True')
                if i-1 > 0:
                    Map[j][i - 2].setObstacle('True')
                if i < (dimX - 1):
                    Map[j][i + 1].setObstacle('True')
                if i+1 < (dimX - 1):
                    Map[j][i + 2].setObstacle('True')
                # Diag values
                if j > 0 and i > 0:                     # top left
                    Map[j - 1][i - 1].setObstacle('True')
                if j > 0 and i < (dimX - 1):            # top right
                    Map[j - 1][i + 1].setObstacle('True')
                if j < (dimY - 1) and i < (dimX - 1):   # bottom right
                    Map[j + 1][i + 1].setObstacle('True')
                if j < (dimY - 1) and i > 0:            # bottom left
                    Map[j + 1][i - 1].setObstacle('True')
                #----------------------------------

            ## Thymio contour
            elif window_Thymio.mean()>127:
                Map[j][i].setGround('unknown')

            else:               # Else: no obstacle nor Thymio have been detected
                                # We have then to determine the color of the ground

                window_bw = rescaled_bw[ (j*edgeY):((j+1)*edgeY) , (i*edgeX):((i+1)*edgeX) ]

                if window_bw.mean() > 0.5:
                    Map[j][i].setGround('white')
                else:
                    Map[j][i].setGround('black')

    # Pattern map (black box = 1, white box = 0)
    # Initialization of Pattern as a matrix of '0'
    Pattern = [[0 for i in range(dimY)] for j in range(dimX)]
    Pattern = np.array(Pattern)
    Pattern = Pattern.reshape(dimY, dimX)
    for j in range(0, dimY):
        for i in range(0, dimX):
            if Map[j][i].getGround() == 'black':      # 'black'   = 1
                Pattern[j][i] = 1
            elif Map[j][i].getGround() == 'unknown':  # 'unknown' = 2
                Pattern[j][i] = 2
            else:                                     # 'white'   = 0
                Pattern[j][i] = 0

    # Fixed obstacles map (obstacle = 1, free = 0)
    # Initialization of Pattern as a matrix of '0'
    Obstacles = [[0 for i in range(dimY)] for j in range(dimX)]
    Obstacles = np.array(Obstacles)
    Obstacles = Obstacles.reshape(dimY, dimX)
    for j in range(0, dimY):
        for i in range(0, dimX):
            if Map[j][i].isObstacle():  # 'isObstacle==True'   = 1
                Obstacles[j][i] = 1
            else:                       # 'isObstacle==False'  = 0
                Obstacles[j][i] = 0

    # Defining the start position of Thymio
    start = [m.floor(start_pxl[0]/edgeX), m.floor(start_pxl[1]/edgeY), start_pxl[2]]

    return Map, Pattern, Obstacles, start

def detect_obstacles(rescaled):
    """
    :brief: detects the location of the obstacles (i.e. their y and x coordinates
    :param rescaled: ndarray containing the rescaled, cropped and adjusted image
    :return mask_obst: ndarray containing the points of location of the obstacles

    requires following package:
        import cv2

    reference:
    [ST]
        STONE, Rebecca, 2019. Image Segmentation Using Color Spaces in OpenCV + Python. Real Python [online].
        2019.12.06. Consulted on 2019.12.04. Available at: https://realpython.com/python-opencv-color-spaces/
    """

    # Image Segmentation Using Color Spaces in OpenCV + Python, cf. [ST]
    # HSV
    hsv_rescaled = cv2.cvtColor(rescaled, cv2.COLOR_RGB2HSV)
    ## Detecting green obstacles
    # Picking out a range
    light_green = (40, 65, 80)
    dark_green = (90, 255, 255)
    # Binary mask ('1s' indicate values within the range, and '0s' values indicate values outside)
    mask_obst = cv2.inRange(hsv_rescaled, light_green, dark_green)

    return mask_obst

def detect_thymio_pose(rescaled):
    """
    :brief: detects the pose of the robot (i.e. its y and x coordinates and its orientation)
    :param rescaled: ndarray containing the rescaled, cropped and adjusted image
    :return mask_Thymio: ndarray containing the points of location of the robot
    :return start_pxl: list containing the pose of the robot in [pixel] and its orientation in degrees

    requires following packages:
        import cv2
        import math as m
        import numpy as np

    reference:
    [ST]
        STONE, Rebecca, 2019. Image Segmentation Using Color Spaces in OpenCV + Python. Real Python [online].
        2019.12.06. Consulted on 2019.12.04. Available at: https://realpython.com/python-opencv-color-spaces/
    """

    # Image Segmentation Using Color Spaces in OpenCV + Python, cf. [ST]
    # HSV
    hsv_rescaled = cv2.cvtColor(rescaled, cv2.COLOR_RGB2HSV)
    # Our detection technique will be based on hsv_rescaled

    ## Detecting rear turquoise dot of robot
    # Picking out a range
    light_blue = (0, 50, 50)
    dark_blue = (30, 255, 255)
    # Binary mask ('1s' indicate values within the range, and '0s' values indicate values outside)
    mask_rear = cv2.inRange(hsv_rescaled, light_blue, dark_blue)
    ### Calculating rear center
    # detecting the index of the pixels forming the rear of the robot
    index_rear = np.where(mask_rear==255)
    if len(index_rear[0]) > 0: # If we have detected the rear of the robot
        # Taking the mean y of all these points
        rear_y = m.floor(index_rear[0].mean())
        # Taking the mean x of all these points
        rear_x = m.floor(index_rear[1].mean())

    ## Detecting red front of robot
    # Picking out a range
    light_red = (110,120,130)
    dark_red = (130,255,255)
    # Binary mask ('1s' indicate values within the range, and '0s' values indicate values outside)
    mask_front = cv2.inRange(hsv_rescaled, light_red, dark_red)
    ## Calculating front center
    # detecting the index of the pixels forming the rear of the robot
    index_front = np.where(mask_front==255)
    if len(index_front[0]) > 0:  # If we have detected the front of the robot
        # Taking the mean y of all these points
        front_y = m.floor(index_front[0].mean())
        # Taking the mean x of all these points
        front_x = m.floor(index_front[1].mean())

    ## Combining rear and front mask of the robot
    mask_Thymio = mask_rear + mask_front

    if ( len(index_front[0]) > 0 and len(index_rear[0]) > 0 ): # If we have detected a Thymio
        ## Calculating Thymio Pose
        # Thymio x, y coordinates
        Thymio_x = rear_x
        Thymio_y = rear_y
        # Thymio orientation
        b = front_y-rear_y
        a = front_x-rear_x
        Thymio_theta = m.atan2(b,a)*180/m.pi

        # Start coordinates of Thymio (in [pxl])
        start_pxl = ([Thymio_x,Thymio_y,Thymio_theta])
    else:
        start_pxl = ([0,0,0])

    return mask_Thymio, start_pxl

def vision(path, dimX, dimY):
    """
    :param path: the path to the base picture of the map
    :param dimX: number of boxes on the width of our rescaled picture
    :param dimY: number of boxes on the height of our rescaled picture
    :return Map: ndarray of class MapNode containing information about the color of the ground and
            the presence of obstacle for every coordinate
    :return Pattern: ndarray representing the pattern on the ground (0 = white, 1 = black, 2 = unknown)
    :return Obstacles: ndarray representing the obstacles (0 = free areas, 1 = obstacles)
    :return start: list containing the pose of the robot according to the coordinate of the discretized map and
            its orientation in degrees

    requires following packages:
        import cv2
        import imageio
        import math as m
        import numpy as np
        import os
        from PIL import Image
        from scipy import signal
    """
    output_sobel, rescaled, rescaled_bw = rectify_map(path)
    mask_Thymio, start_pxl = detect_thymio_pose(rescaled)
    mask_obst = detect_obstacles(rescaled)
    Map, Pattern, Obstacles, start = remapping(rescaled_bw, mask_obst, mask_Thymio, dimX, dimY, start_pxl)

    return Map, Pattern, Obstacles, start