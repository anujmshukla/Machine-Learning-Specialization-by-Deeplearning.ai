import copy, math
import numpy as np
import matplotlib.pyplot as plt

def Predict(x, w, b): 
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b     
    return p    


def Compute_Cost(x,y,w,b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    
    sum_Cost = 0
    sizeOfInput = x.shape[0] #return how many rows are there in data
    for i in range(0,sizeOfInput):
        f_wb = np.dot(x[i],w) + b
        cost = (f_wb - y[i]) ** 2
        sum_Cost += cost
    
    total_cost = (1/(2*sizeOfInput))*sum_Cost
    return total_cost

def Compute_Gradient(x,y,w,b):
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    
    Num_of_rows, Num_of_columns = x.shape 
    sum_dj_dw = 0
    sum_db_dw = 0
    
    for i in range(0,Num_of_rows):
        f_wb = np.dot(x[i],w) + b
        diffrence = f_wb - y[i]
        sigmaTermFor_dj_dw = diffrence*x[i]
        sum_dj_dw += sigmaTermFor_dj_dw
        sum_db_dw += diffrence
        
    dj_dw = (1/Num_of_rows)*sum_dj_dw
    dj_db = (1/Num_of_rows)*sum_db_dw
    
    return dj_dw,dj_db

def Compute_Gradient_Descent(x, y, w_in, b_in, iteration, alpha, ):
    
    cost_function = Compute_Cost
    Gradient_Descent = Compute_Gradient
    
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      J_History        : History of cost function changes with iteration 
    """
    J_History = []
    
    for i in range(0,iteration):
        
        #getting Gradient
        dj_dw,dj_db = Gradient_Descent(x,y,w_in,b_in)
        
        w_in = w_in - alpha*dj_dw  # "w_in" is an array so "- alpha*dj_dw" operation will perfrom on all elements of an array 
        b_in = b_in - alpha*dj_db
        
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_History.append( cost_function(x, y, w_in, b_in))
            
        #print cost after each 100 iterarion 
        if(i%100==0):
            print(f"Iteration {i}: Cost {J_History[-1]}")
            
    return w_in, b_in, J_History
            