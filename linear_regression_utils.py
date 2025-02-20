import numpy as np
import numpy.typing as npt

def calculate_cost(x: npt.NDArray[np.number], y: npt.NDArray[np.number], w, b):
    '''
    returns the cost function for linear regression: J(w,b)

    parameters:
    - x: vector with x values training_data, m examples. ndarray(m,)
    - y: vector with y values target_values m examples. ndarray(m,)
    - w: slope of linear regression. scalar value
    - b: intercept of linear regression. scalar value
    '''

    #calculate number of examples:
    m = x.shape[0]
    #check valid sample size:
    if m<=0:
        return None

    #initialize variables
    cost_sum=0
    
    #iterate thorugh examples. Aware that could do vetorized sum through numpy, but I want to implement from scratch to
    # learn
    for i in range(m):
        f_wb = w*x[i] + b
        cost = (f_wb - y[i])**2
        cost_sum += cost
    
    #calculate based on cost function formula:
    total_cost = (1/(2*m))*cost_sum
    return total_cost

def calculate_gradient(x: npt.NDArray[np.number], y: npt.NDArray[np.number], w, b):
    '''
    returns both partial derivatives from gradient descent equation: returns dj/dw and dj/db
        used as part of gradient descent calculus

    parameters:
    - x: vector with x values training_data, m examples. nparray(m,)
    - y: vector with y values target_values, m examples. nparray(m,)
    - w: slope of linear regression. scalar value
    - b: intercept of linear regression. scalar value
    '''
    #define sample size:
    m = x.shape[0]
    #validate sample size:
    if m <= 0:
        raise ValueError('Invalid sample size. Check x vector')

    #initialize variables
    dw_sum = 0
    db_sum = 0

    #iterate through examples. Aware that could do vetorized sum through numpy, but I want to implement from scratch to
    # learn:
    for i in range(m):
        f_wb = w*x[i] + b
        dw_iteration = (f_wb - y[i])*x[i]
        db_iteration = (f_wb - y[i])
        dw_sum += dw_iteration
        db_sum += db_iteration

    dj_dw = dw_sum/m
    dj_db = db_sum/m

    return dj_dw, dj_db

def calculate_gradient_descent(x_vector: npt.NDArray[np.number] , y_vector: npt.NDArray[np.number], w_initial, b_initial,
                               num_iterations, alpha, generate_history=False):
    '''
    returns w and b values calulate bu gradient descent algorithm. In case generate_history is selected, additionaly 
    outputs params values history and cost function values.

    parameters:
    - x_vector: vector with x values training_data, m examples. nparray(m,)
    - y_vector: vector with y values target_values, m examples. nparray(m,)
    - w_initial: initial chosen value for slope of linear regression. scalar value
    - b_initial: initial chosen value for intercept of linear regression. scalar value
    - num_iterations: number of iterations for gradient descent. scalar value
    - alpha: learning rate for gradient descent algorithm. scalar value
    - generate_history: selector for generating two list of values while iterating
    '''
    #initiate w and b
    w = w_initial
    b = b_initial

    if generate_history:
        params_history = []
        j_history = []
        
        for _ in range(num_iterations):
            #define partial derivatives
            dj_dw, dj_db = calculate_gradient(x_vector, y_vector, w, b) #returns partial derivatives for both w and b

            #calculates new w and b values according to gradient descent algorithm
            w_temp = w - alpha*dj_dw
            b_temp = b - alpha*dj_db

            #simultaneously update w and b values	
            w = w_temp
            b = b_temp

            #register history for plotting graphs and other utilities
                #limit max number of registers to prevent resurce exhaustion. Register every 10 iterations do avoid overregistry
            if (num_iterations<1000) and (num_iterations%10==0): 
                params_history.append([w,b])
                j_history.append(calculate_cost(x_vector, y_vector, w, b))
        return w, b, params_history, j_history
    else:
        for _ in range(num_iterations):
            #define partial derivatives
            dj_dw, dj_db = calculate_gradient(x_vector, y_vector, w, b) #returns partial derivatives for both w and b

            #calculates new w and b values according to gradient descent algorithm
            w_temp = w - alpha*dj_dw
            b_temp = b - alpha*dj_db

            #simultaneously update w and b values	
            w = w_temp
            b = b_temp

        return w,b


