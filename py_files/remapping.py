import math as m
import numpy as np

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