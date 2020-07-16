import math as m
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def create_empty_plot(size_X,size_Y):
    """
    Helper function to create a figure of the desired dimensions & grid
    
    :params size_X: dimension of the map along the x dimension
    :params size_Y: dimension of the map along the y dimension
    :return: the fig and ax objects.
    """
    max_val=max(size_X,size_Y)
    fig, ax = plt.subplots(figsize=(7,7))
    
    major_ticks = np.arange(0, max_val+1, 5)
    minor_ticks = np.arange(0, max_val+1, 1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.set_ylim([size_Y,-1])
    ax.set_xlim([-1,size_X])
    ax.grid(True)
    
    return fig, ax


def PlotInitialMap(Obstacles, start):
    """ 
    Build the plot of the map with Obstcales and Thymio initial pose
    :param Obstacles: The array containing the obstacle map
    :param start: The array containing X,Y coordinates and orientation of Thymio 
    :return: the fig and ax object of the initial map
    """
    #StartCoordinates=(start[0], start[1])   #Defines the robot initial position for Display and A* algorithm
    StartThymioAngle=-m.radians(start[2])

    fig_map, ax_map = create_empty_plot(Obstacles.shape[1],Obstacles.shape[0])
    cmap = colors.ListedColormap(['white', 'red']) # Select the colors with which to display obstacles and free cells

    ax_map.imshow(Obstacles, cmap=cmap)
    ax_map.plot(start[0], start[1], marker="p", color = 'green', markersize=12)
    ax_map.quiver(start[0], start[1], m.cos(StartThymioAngle), m.sin(StartThymioAngle), scale_units='height', scale=7, width=0.005, color = 'green') 
    
    return fig_map, ax_map


def PlotPathMap(Obstacles, startCoordinateXY, goal, path, visitedNodes):
    """ 
    Build the plot of the map with path, Obstcales and Thymio initial pose
    :param Obstacles: The array containing the obstacle map
    :param startCoordinateXY: The array containing X,Y coordinates of Thymio 
    :param goal: The array containing the final position of Thymio
    :param path: The array containing the optimal path of Thymio 
    :param vistedNodes: The array of explored nodes by the A* algorithm
    :return: the fig and ax object of the optimal path plot
    """
    # Reshape path
    path = path.transpose()
    # Displaying the map
    size_X=Obstacles.shape[1]
    size_Y=Obstacles.shape[0]
    fig_astar, ax_astar = create_empty_plot(size_X,size_Y)
    cmap = colors.ListedColormap(['white', 'red']) # Select the colors with which to display obstacles and free cells

    ax_astar.imshow(Obstacles, cmap=cmap)

    # Plot the best path found and the list of visited nodes
    ax_astar.scatter(visitedNodes[0], visitedNodes[1], marker="o", color = 'orange');
    ax_astar.plot(path[0], path[1], marker="o", color = 'blue');
    ax_astar.scatter(startCoordinateXY[0], startCoordinateXY[1], marker="o", color = 'green', s=200);
    #p1=ax_astar.scatter(goal[0], goal[1], marker="s", color = 'yellow', s=300, alpha=0.88);

    #import matplotlib.image as mpimg
    emoji = plt.imread('images/1f3c1.png')
    imagebox = OffsetImage(emoji, zoom=0.5, alpha=0.9)
    ArrivalFlag = AnnotationBbox(imagebox, (goal[0], goal[1]), frameon=False, box_alignment=(0.1,0.1))

    ax_astar.add_artist(ArrivalFlag)
    
    return fig_astar, ax_astar