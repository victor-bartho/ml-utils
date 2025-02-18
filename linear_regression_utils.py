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
    - y: vector with y values target_values m examples. nparray(m,)
    - w: slope of linear regression. scalar value
    - b: intercept of linear regression. scalar value
    '''
    #define sample size:
    m = x.shape[0]
    #validate sample size:
    if m <= 0:
        return None

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