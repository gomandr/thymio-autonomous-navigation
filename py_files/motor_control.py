import math as m
import time

def motor_stop():
    th.set_var("motor.left.target", 0)
    th.set_var("motor.right.target", 0) 
    
def motor_forward(motor_speed):
    th.set_var("motor.left.target", motor_speed)
    th.set_var("motor.right.target", motor_speed)
    
def motor_rotate(direction,motor_rot):

    if direction == "left":
        th.set_var("motor.right.target",motor_rot)
        th.set_var("motor.left.target",2**16 - motor_rot)
        
    if direction == "right":
        th.set_var("motor.left.target",motor_rot)
        th.set_var("motor.right.target",2**16 - motor_rot)  
        
def global_controller(state,theta,reference,motor_speed,motor_rot,coeff_rot):
    dist_vector = reference-state # vector between state and reference
    theta_ref = m.atan2(dist_vector[1],dist_vector[0]) # Vector angle from x-axis 
    dtheta = normalise_angle(theta_ref - theta) # Normalize angle to minimise rotation (270deg -> -90°)
        
    if abs(dtheta) > m.pi/16:
        theta = theta + dtheta
        
        if dtheta < 0: # Opposite from convention
            motor_rotate("left",motor_rot)
        else:
            motor_rotate("right",motor_rot)
            
        time.sleep(abs(coeff_rot*dtheta/(2*m.pi)))# 8.246 [s] for 2pi rad turn
        
        motor_stop()  
    
    print("Turning angle: ", dtheta*180/m.pi)
    
    motor_forward(motor_speed)
    
    return theta

def rot_angle(Ts,coeff_rot):
    dtheta = Ts*2*m.pi/coeff_rot
    return dtheta