# Basics of mobile robotics - Thymio Project - December 2019

### Students : Colomban Fussinger, André Gomes, Anthony Guinchard, Sylvain Pietropaolo

The goal of this project is to combine vision, path planning, local navigation and filtering to maneuver a Thymio robot on a map around obstacles. First a picture of the map is taken where the robot will navigate. From this picture, the map pattern is saved, the obstacles are highlighted and the start position and orientation are shown on a plot in order to help to select the goal. Once chosen the algorithm computes the optimal path and send the instructions to the Thymio robot. The robot sends back to the computer the odometry and the colors seen under his bodyshell but also indicates the presence of obstacle in front of him. These informations are used by the computer to compute the position of the robot and then to give instructions to the motors to follow the optimal path and to avoid the obstacles.


### Video of the project
https://www.youtube.com/watch?v=j_90MK3PXCc
