import cv2

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