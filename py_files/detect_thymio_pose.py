import math as m
import numpy as np
import cv2

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