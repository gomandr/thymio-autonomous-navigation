import math as m
import numpy as np

def target(pos,optimal_path,k):
    """
   Set new reference to follow
    
    :param:
                pos, position (2x1) (first 2 rows of state estimate)
                global_path, optimal path from A* alg (2xn)
    :return: 
                target, reference position to go to(2x1)
    """          
    #k = 2 # constant for uplooking in the optimal 
                
    # Compute distance between Thymio and each position of the optimal path
    dist = np.linalg.norm(pos-optimal_path, axis=1)      
    # Selects the closest point
    step = np.argmin(dist)
    if step + k < len(optimal_path):
        reference = optimal_path[step+k]
    else:
        reference = optimal_path[-1]
    return reference

def goal_reached():
    motor_stop()
    
def normalise_angle(alpha):
    while alpha > m.pi:
        alpha -= 2. * m.pi
    while alpha < -m.pi:
        alpha += 2. * m.pi
    return alpha