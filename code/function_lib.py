import casadi.casadi as cs

def model(x,y,theta,u,ts):
    # Get the velocities for readability 
    v,w = u[0], u[1]

    # Update according to the model
    x += ts*cs.cos(theta)*v
    y += ts*cs.sin(theta)*v
    theta += ts*w

    return x,y,theta

def generate_straight_trajectory(x,y,theta,v,ts): 
    states = [x,y,theta]

    for i in range(0,19): 
        x,y,theta = model(x,y,theta,[v,0],ts=ts)
        states.extend([x,y,theta])

    return states