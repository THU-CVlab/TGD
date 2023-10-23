import numpy as np
import math

def Exp1D(x, param):
    return param['k'] * np.exp(param['delta'] * -abs(x))

def Landau1D(x, param):
        return param['c'] * (1 - (x**2 / (param['r']**2)))**param['n']

def Linear1D(x, param):
        if x > 0:
            return param['k'] * x - (param['b'])
        else:
            return param['k'] * x + (param['b'])
def Weibull1D(x, param):
    if x < 0:
        return param['k']/param['lambda'] * (-x/param['lambda'])**(param['k']-1)*np.exp(-(-x / param['lambda']) ** param['k'])
    elif x == 0 and param['k'] < 1:
        return 0
    else:
        return -param['k']/param['lambda'] * (x/param['lambda'])**(param['k']-1)*np.exp(-(x / param['lambda']) ** param['k'])

# one-dimensional Gaussian function
def Gauss1D(x, param):
    return ((1/((math.sqrt(2*math.pi))*param["delta"])) * math.exp(-(x**2)/(2*(param["delta"]**2))))

# 2D Gaussian function
def Gauss2D(x, y, param):
    return (1/(2 * math.pi * (param["delta"]**2)) * math.exp(-(x**2 + y**2)/(2*(param["delta"]**2))))
    
# One-dimensional Gaussian function derivative
def Gauss1D_Derivative(x, param):
    return (-1 * x * (1/((math.sqrt(2*math.pi))*(param["delta"]**3))) * math.exp(-(x**2)/(2*(param["delta"]**2))))

# One-dimensional Gaussian second-order derivative function
def Gauss1D_Derivative2(x, param):
    return (((x**2 - param["delta"]**2)/((math.sqrt(2*math.pi))*(param["delta"]**4))) * math.exp(-(x**2)/(2*(param["delta"]**2))))

# First-order derivatives of a two-dimensional Gaussian
def Gauss2D_Derivative(x, y, param):
    return abs(Gauss2D_Derivative_X(x, y, param["delta"])) + abs(Gauss2D_Derivative_Y(x, y, param["delta"]))
    
# Directional derivatives of a two-dimensional Gaussian
def Gauss2D_Derivative_L(x, y, param, theta):
    return Gauss2D_Derivative_X(x, y, param["delta"]) * math.cos(theta) + Gauss2D_Derivative_Y(x, y, param["delta"]) * math.sin(theta)

# 2D Gaussian dG/dx
def Gauss2D_Derivative_X(x, y, param):
    return (-1 * x * (1/(2*math.pi*(param["delta"]**4))) * math.exp(-(x**2+y**2)/(2*(param["delta"]**2))))
    
# 2D Gaussian dG/dy
def Gauss2D_Derivative_Y(x, y, param):
    return (-1 * y * (1/(2*math.pi*(param["delta"]**4))) * math.exp(-(x**2+y**2)/(2*(param["delta"]**2))))

# Laplace of Gauss
def Gauss2D_Laplace(x, y, param):
    return ((-1 * (1 / (2 * math.pi * (param["delta"]**4))) * (2 - (x**2 + y**2) / (param["delta"]**2))) * math.exp(-(x**2+y**2)/(2*(param["delta"]**2))))

# One-dimensional smooth convolution kernel
def GaussSmooth_1D(width, step, param, normalization = True):
    n = (int)(width / step)
    x_list = [step * i for i in range(-n, n + 1, 1)]
    gauss_smoothing_kernel = np.array([Gauss1D(x, param) for x in x_list])
    if normalization:
        gauss_smoothing_kernel /= gauss_smoothing_kernel.sum()  # 归一化
    return (x_list, gauss_smoothing_kernel)

# One-dimensional first-order convolution kernel
def GaussDerivative1_1D(width, step, param):
    n = (int)(width / step)
    x_list = [step * i for i in range(-n, n + 1, 1)]
    x_list_left = [step * i for i in range(-n, 0, 1)]
    x_list_right = [-step * i for i in range(1, n+1, 1)]
    gauss_derivative_kernel = [Gauss1D_Derivative(x, param) for x in x_list_left] + [0] + [-Gauss1D_Derivative(x, param) for x in x_list_right]
    gauss_derivative_kernel = np.array(gauss_derivative_kernel)
    return (x_list, gauss_derivative_kernel)

# One-dimensional second-order convolution kernel
def GaussDerivative2_1D(width, step, param):
    n = (int)(width / step)
    x_list = [step * i for i in range(-n, n + 1, 1)]
    gauss_derivative_kernel = np.array([Gauss1D_Derivative2(x, param) for x in x_list])

    sum_res1 = abs(gauss_derivative_kernel).sum()
    sum_res2 = gauss_derivative_kernel.sum()
    negative_sum = (sum_res1 - sum_res2) / 2
    positive_sum = (sum_res1 + sum_res2) / 2

    # Convolution kernel sums to 0
    for r in range(2 * n + 1):
        if gauss_derivative_kernel[r] > 0:
            gauss_derivative_kernel[r] /= positive_sum
        else:
            gauss_derivative_kernel[r] /= negative_sum
        # gauss_derivative_kernel[int(width / step)] = 0
        # gauss_derivative_kernel[int(width / step)] = -gauss_derivative_kernel.sum()      # 卷积核求和为0
    return (x_list, gauss_derivative_kernel)

# 2D Smooth Convolution Kernel
def GaussSmooth_2D(width, step, param, normalization = True):

    n = (int)(width / step)
    pos_x = np.linspace(-n * step, n * step, 2 * n + 1)
    pos_y = np.linspace(-n * step, n * step, 2 * n + 1)
    xv, yv = np.meshgrid(pos_x, pos_y)
    res = np.zeros((2 * n + 1, 2 * n + 1))
    for r in range(2 * n + 1):
        for c in range(2 * n + 1):
            x = xv.item((r,c))
            y = yv.item((r,c))
            res[r][c] = Gauss2D(x, y, param)
    if normalization:
        res /= res.sum()  # normalization
    ret = (xv, yv, res)
    return ret

# Two-dimensional first-order convolution kernel
def GaussDerivative1_2D(width, step, param, theta):

    n = (int)(width / step)
    pos_x = np.linspace(-n * step, n * step, 2 * n + 1)
    pos_y = np.linspace(-n * step, n * step, 2 * n + 1)
    xv, yv = np.meshgrid(pos_x, pos_y)
    res = np.zeros((2 * n + 1, 2 * n + 1))
    for r in range(2 * n + 1):
        for c in range(2 * n + 1):
            x = xv.item((r,c))
            y = yv.item((r,c))
            if theta == 0:
                org_x = x
                org_y = y
            elif theta == 90:
                org_x = y
                org_y = -x
            else:
                theta_ = math.pi / (180 / theta)
                org_x = x * math.cos(-theta_) - y * math.sin(-theta_)
                org_y = x * math.sin(-theta_) + y * math.cos(-theta_)
            res[r][c] = Gauss2D_Derivative_X(org_x, org_y, param)

    res /= res.max()
    ret = (xv, yv, res)
    return ret


# Two-dimensional second-order convolution kernel
def LoG_2D(width, step, param):

    n = (int)(width / step)
    pos_x = np.linspace(-n * step, n * step, 2 * n + 1)
    pos_y = np.linspace(-n * step, n * step, 2 * n + 1)
    xv, yv = np.meshgrid(pos_x, pos_y)

    if width == 1:
        res = np.array([[1,1,1],[1,-8,1],[1,1,1]]) / 8
    elif width == 2:
        res = np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]]) / 16
    else:
        res = np.zeros((2 * n + 1, 2 * n + 1))
        for r in range(2 * n + 1):
            for c in range(2 * n + 1):
                x = xv.item((r, c))
                y = yv.item((r, c))
                res[r][c] = Gauss2D_Laplace(x, y, param)

        sum_res1 = abs(res).sum()
        sum_res2 = res.sum()
        negative_sum = (sum_res1 - sum_res2) / 2
        positive_sum = (sum_res1 + sum_res2) / 2

        for r in range(2 * n + 1):
            for c in range(2 * n + 1):
                if res[r][c] > 0:
                    res[r][c] /= positive_sum
                else:
                    res[r][c] /= negative_sum

        res /= abs(res[n][n])

        # res[n][n] = res[n][n] - res.sum()       # Reduce the error by ensuring that all elements of the kernel have a sum or mean value of zero.
        # res /= abs(res[n][n])                   # normalization
    # print('LoG kernel sum: ', res.sum())
    ret = (xv, yv, res)
    return ret

def Sobel3x3(theta):
    assert int(theta) in [0,45,90,135]

    pos_x = np.linspace(-1, 1, 2)
    pos_y = np.linspace(-1, 1, 2)
    xv, yv = np.meshgrid(pos_x, pos_y)

    sobel_0   = np.array([[1,0,-1], [2,0,-2],[1,0,-1]])
    sobel_45  = np.array([[2,1,0], [1,0,-1],[0,-1,-2]])
    sobel_90  = np.array([[2,1,0], [1,0,-1],[0,-1,-2]])
    sobel_135 = np.array([[0,1,2], [-1,0,1],[-2,-1,0]])

    
    if theta == 0:
        ret = (xv, yv, sobel_0)
    elif theta == 45:
        ret = (xv, yv, sobel_45)  
    if theta == 90:
        ret = (xv, yv, sobel_90)
    elif theta == 135:
        ret = (xv, yv, sobel_135)
    return ret

