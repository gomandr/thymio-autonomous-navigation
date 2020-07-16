
def speed_measurement(motor_speed,coeff_forward):
    # Measures motors speed
    speed_left = th["motor.left.speed"]
    speed_right = th["motor.right.speed"]
    
    # Take the mean value
    motor_mean = np.mean([speed_left, speed_right])
    
    # Converts to unit of grid per second
    if motor_mean > motor_speed*1.1:
        speed = motor_speed*coeff_forward
    else:
        speed = motor_mean*coeff_forward    
                     
    return speed

def ground_measurement(ground_sensed):
    
    ground_values = th["prox.ground.delta"]
    
    # Update last measurement
    last_ground_sensed = ground_sensed
    
    # If color sensed is white -> set new measurement to 0
    if ground_values[0] > 500:
        ground_sensed = 0
    else:
        ground_sensed = 1
        
    return last_ground_sensed, ground_sensed        