import math as m
import numpy as np
from random import sample
from numpy import ndarray

def mcl(speed, Ts, ground_truth, last_ground_sensed, ground_sensed, particles, theta, nb_particles):
    """
    Returns the state estimate (position 2D and velocities 2D) using Monte Carlo Localisation algorithm.
    Discrete LTI system: x+ = Ax + Bu
    with x = [posX,posY] and speed = motorspeeds
    
    :param:
                ground_truth, binary mapping of the ground (NxN)
                particles, particles state from previous iteration (Mx3)
                groud_sensed, binary value from the ground sensor (1x1)
                speed (1x1), motor speed
                Ts, time since last iteration (1x1)
                nb_particles, number of particles for the filter (1x1)
    :return: 
                particles, updated particles (Mx3)
    """
    
    nb_rand_particles = 0
    cov = [[0.03*Ts,0],[0,0.3*Ts]]
    tot_particles = nb_particles + nb_rand_particles
    CM_to_sensor = [2.33*m.cos(theta)+0.25*m.sin(theta), 2.33*m.sin(theta)-0.25*m.cos(theta)] # 7 cm for 3 cm grid
    x_max = 30 
    y_max = 20
    
    # Add random particles to the set
    rand_particles = np.random.rand(nb_rand_particles,2)
    rand_particles[:,0] = rand_particles[:,0]*x_max
    rand_particles[:,1] = rand_particles[:,1]*y_max
    particles = np.concatenate((particles,rand_particles),axis=0)
    
    # Converts array to list
    particles = particles.tolist()
    
    # Create an empty temporary set
    particles_tilde = ndarray((tot_particles,2),float)    
    w = np.empty(tot_particles)
    xi = np.empty(2)
    for i in range(tot_particles):
    
        # Sample one particle from the set without replacement
        xi = sample(particles,1)
        xi = xi[0]
        # Update position of particule based on the model
        xi[0] = xi[0] + m.cos(theta)*speed*Ts
        xi[1] = xi[1] + m.sin(theta)*speed*Ts
        
        # Select a particle on a gaussian distribution around updated position
        xi = np.random.multivariate_normal(xi, cov, 1)
        xi = xi[0]
        
        # Compute weight associated to position and add it to state
        # Points outside map are white == 0
        if xi[0]+CM_to_sensor[0] < -0.5 or xi[0]+CM_to_sensor[0] > x_max+0.49 or \
        xi[1]+CM_to_sensor[1] < -0.5 or xi[1]+CM_to_sensor[1] > y_max+0.49 :
            if ground_sensed == 0:
                w[i] = 0.9
            else :
                w[i] = 0.1
                
        else:
            # Get true value of the ground at the updated position
            xi_flat = np.round([xi[0]+CM_to_sensor[0], xi[1]+CM_to_sensor[1]])
            xi_ground_truth = ground_truth[int(xi_flat[0]),int(xi_flat[1])]
            if xi_ground_truth == ground_sensed :
                w[i] = 0.9
            else :
                w[i] = 0.1
        
        # Add updated particule to the temporary set
        particles_tilde[i] = xi
        
    # Recover weigths
    w = w/sum(w)
    # Sample randomly from the temporary set according to the weigths
    particles = particles_tilde[np.random.choice(particles_tilde.shape[0], nb_particles, replace=True, p=w),0:2]

    return(particles)
    
def estimate_state(particles):    
    # limits for considering participating to the state estimation
    xy_lim = 1.5

    # RANSAC to find best index
    iterations_count = 500
    tests_count = 500
    index = -1
    o_index = -1
    best_index = -1
    support = 0
    best_support = 0
    particles_view = np.asarray(particles)
    max_index = particles_view.shape[0]-1
    iteration_indices = np.random.randint(0, max_index, [iterations_count])
    test_indices = np.random.randint(0, max_index, [tests_count])

    # tries a certain number of times
    for i in range(iterations_count):
        index = iteration_indices[i]
        x = particles_view[index, 0]
        y = particles_view[index, 1]
        support = 0
        for j in range(tests_count):
            o_index = test_indices[j]
            o_x = particles_view[o_index, 0]
            o_y = particles_view[o_index, 1]
            # compute distance
            dist_xy = m.sqrt((x-o_x)*(x-o_x) + (y-o_y)*(y-o_y))
            if dist_xy < xy_lim:
                support += 1
        # if it beats best, replace best
        if support > best_support:
            best_index = index
            best_support = support
    # then do the averaging for best index
    x = particles_view[best_index, 0]
    y = particles_view[best_index, 1]
    xs = 0.
    ys = 0.
    count = 0
    #conf_count = 0
    for j in range(tests_count):
        o_index = test_indices[j]
        o_x = particles_view[o_index, 0]
        o_y = particles_view[o_index, 1]
        dist_xy = m.sqrt((x-o_x)*(x-o_x) + (y-o_y)*(y-o_y))
        if dist_xy < xy_lim:
            xs += o_x
            ys += o_y
            count += 1
    assert count > 0, count
    x_m = xs / count
    y_m = ys / count
    return np.array([x_m, y_m])

def print_particles(particles):
    plt.cla()
    plt.clf()
    plt.close()
    # create data
    x = particles[:,0]
    y = particles[:,1]

    # Big bins
    h,_, _, _ = plt.hist2d(x, y, range=[[0, 30], [0, 20]], bins=(30, 20), cmap=plt.cm.BuPu)

    # Limits
    plt.axvline(0, color='black')
    plt.axhline(0, color='black')
    plt.axvline(30, color='black')
    plt.axhline(20, color='black')

    # State
    plt.axvline(state[0], color='r')
    plt.axhline(state[1], color='green')

    plt.colorbar()
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.gca().invert_yaxis()
    plt.show()