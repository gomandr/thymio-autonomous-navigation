def local_nav(sens,treshold_side,treshold_obst,time_forward,motor_speed,motor_rot,particles,ground_truth,ground_sensed,theta,nb_particles):
    
    # follow the obstacle by the left
    if (sens[0]+sens[1]) > (sens[4]+sens[3]) :
        state, theta, particles = side_by_left(sens,treshold_side,time_forward,motor_speed,motor_rot,particles,ground_truth,ground_sensed,theta,nb_particles)
    # follow the obstacle by the right    
    else :
        state, theta, particles = side_by_right(sens,treshold_side,time_forward,motor_speed,motor_rot,particles,ground_truth,ground_sensed,theta,nb_particles)
    
    return state, theta, particles
        
def side_by_left(sens,treshold_side,time_forward,motor_speed,motor_rot,particles,ground_truth,ground_sensed,theta,nb_particles) :
    c = 0  #to start the timer only the first time one enter into the loop
    
    # turn right until the sensor don't see anything
    while sum(sens[i] > treshold_side for i in range(0,len(sens)-2)) > 0 :
        
        motor_rotate("right",motor_rot)
        
        # Set turn-right timer
        if c == 0 :
            t_start = time.perf_counter()
            c=1
    
        sens = th["prox.horizontal"]
        if sum(sens[i] > treshold_side for i in range(0,len(sens)-2)) == 0:
            # Stop turn-right timer
            t_stop = time.perf_counter()
            T_l = t_stop - t_start
            
            # Computes new angle
            dtheta = rot_angle(T_l,coeff_rot)
            theta = theta+dtheta
            
            # Goes forward
            Ts, speed = local_forward(motor_speed,time_forward,treshold_side)
            
            # Measures ground and computes new state
            last_ground_sensed, ground_sensed = ground_measurement(ground_sensed)
            particles = mcl(speed, Ts, ground_truth, last_ground_sensed, ground_sensed, \
                        particles, theta, nb_particles)
            state = estimate_state(particles)

    return state, theta, particles

def side_by_right(sens,treshold_side,time_forward,motor_speed,motor_rot,particles,ground_truth,ground_sensed,theta,nb_particles) :
    c=0  #to start the timer only the first time one enter into the loop
    
    # turn left until the sensor don't see anything
    while sum(sens[i] > treshold_side for i in range(0,len(sens)-2)) > 0 :
        
        motor_rotate("left",motor_rot)
        
        # Set turn-left timer
        if c == 0 :
            t_start = time.perf_counter()
            c=1
    
        sens = th["prox.horizontal"]
        if sum(sens[i] > treshold_side for i in range(0,len(sens)-2)) == 0:
            # Stop turn-left timer
            t_stop = time.perf_counter()
            T_r = t_stop - t_start
            
            dtheta = rot_angle(T_r,coeff_rot)
            theta = theta-dtheta
            
            # Going straight if any other obstacle is encountered
            Ts, speed = local_forward(motor_speed,time_forward,treshold_side)
            last_ground_sensed, ground_sensed = ground_measurement(ground_sensed)
            particles = mcl(speed, Ts, ground_truth, last_ground_sensed, ground_sensed, \
                        particles, theta, nb_particles)
            state = estimate_state(particles)

    return state, theta, particles

def local_forward(motor_speed,time_forward,treshold_close): 
    # Set motors speed to go forward
    speed = 0
    motor_forward(motor_speed)
    
    # Increment time counter while checking for obstacles
    step_forward = time_forward/50
    nb_step = 0
    for i in range (50):
        sens=th["prox.horizontal"]
        if sum(sens[i] > treshold_close for i in range(0,len(sens)-2)):
            break    
        time.sleep(time_forward/50)
        speed = speed + speed_measurement(motor_speed,coeff_forward)
        nb_step = nb_step + 1
    # Stop motors
    motor_stop()
    
    # Compute time forward and average speed
    Ts = nb_step*step_forward
    if nb_step != 0:
        speed = speed/nb_step

    return Ts, speed