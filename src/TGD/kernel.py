from .functions import *
from .utils  import Interpolation, integral
definedKernelFunc = ['Gaussian', 'Linear', 'Exp', 'Landau', 'Weibull']
        
definedParams = {
    'Gaussian': ['delta'],
    'Linear'  : ['k', 'b'],
    'Exp'     : ['k', 'delta'], 
    'Landau'  : ['c', 'n', 'r'],
    'Weibull' : ['k', 'lambda']
}
funcMap1D = {
    'Gaussian': Gauss1D,
    'Linear'  : Linear1D,
    'Exp'     : Exp1D, 
    'Landau'  : Landau1D,
    'Weibull' : Weibull1D
}

class TGDkernel(object):
    def __init__(self, param = None,width = 1.0,kernel_size = 5, kernel_func = 'Gaussian', 
                 normalization = True):

        assert kernel_func in definedKernelFunc
        if param is not None:
            for p in definedParams[kernel_func]:
                assert param[p]


    def getKernel1D(self, param = None, radius = 1.0, kernel_size = 5, kernel_func = 'Gaussian', 
                 derivation_order = 1, normalization = True, smooth = False):

        assert kernel_func in definedKernelFunc
        assert int(derivation_order) in [1,2 ] ,  'Only support first and second derivation now!'
        assert kernel_size %2 == 1
        if param is not None:
            for p in definedParams[kernel_func]:
                assert param[p]
        n = kernel_size // 2 
        step = radius / n
        func   = funcMap1D[kernel_func]
        
        x_list = [step * i for i in range(-n, n + 1, 1)]
        x_left = [step * i for i in range(-n, 0, 1)]

        kernel_left = [func(x, param) for x in x_left]

        if normalization:
            t = 0
            for x in kernel_left: t += x
            kernel_left = [ x / abs(t) for x in kernel_left]

        if derivation_order == 1 :
            kernel_right = [-1. * x for x in kernel_left[: :-1] ]
            kernel = kernel_left + [0] + kernel_right
        elif derivation_order == 2 : 
            # kernel = kernel_left + [-2 * sum(kernel_left) * step] + kernel_left[: :-1]
            kernel = kernel_left + [-2 * sum(kernel_left) ] + kernel_left[: :-1]
            # print(sum(kernel))
        kernel = np.array(kernel,'float32')

        return (x_list, kernel)
    
    def getKernel2D(self,paramx, paramy,theta, method = 'directional',
                    radius = 1.0, kernel_size = 5, kernel_func = 'Gaussian', 
                 derivation_order = 1,  interp_step = 0.1):
        
        n    = kernel_size // 2
        step = radius / n
        func = funcMap1D[kernel_func]
        theta = math.pi * theta /180.0 
        x = np.linspace(-n * step, n * step, 2 * n + 1)
        y = np.linspace(-n * step, n * step, 2 * n + 1)
        xv, yv = np.meshgrid(x, y)
        kernel = np.zeros((2 * n + 1, 2 * n + 1))

        if method == 'directional':
            if derivation_order == 1:
                tx, ty = self.getKernel1D(radius=radius, kernel_size= 2* int(radius/interp_step) +1, 
                derivation_order = 1,param=paramx, kernel_func = kernel_func)
                interp_func_x = Interpolation(tx,ty)

                tn    = n * 50
                tstep = step /25.0
                tx = [-tstep * i for i in range(tn, 0, -1)] + [0]
                ty = np.array([func(x, paramy) for x in tx],'float32')
                ty = list(integral(ty, tstep))
                tx.reverse()
                tx = list(map(abs, tx))
                ty.reverse()
                interp_func_y  = Interpolation(tx,ty)
            elif derivation_order == 2:
                tx, ty = self.getKernel1D(radius=radius*2, kernel_size= 2* int(radius/interp_step) +1, 
                    derivation_order = 2,param=paramx, kernel_func = kernel_func)
                interp =  Interpolation(tx,ty)    
        elif method == 'rotational':
            if derivation_order == 1:
                tx, ty = self.getKernel1D(radius=radius*2, kernel_size= 2* int(radius/interp_step) +1, 
                    derivation_order = 1,param=paramx, kernel_func = kernel_func)
                interp = Interpolation(tx,ty)  
        
            elif derivation_order == 2:
                tx, ty = self.getKernel1D(radius=radius*2, kernel_size= 2* int(radius/interp_step) +1, 
                    derivation_order = 2,param=paramx, kernel_func = kernel_func)
                print('a: ',sum(ty))
                interp =  Interpolation(tx,ty)  

        for i in range(2 * n + 1):
            for j in range(2 * n + 1):
                x = xv[i,j]
                y = yv[i,j]
                r = np.sqrt(x**2 + y**2)
                
                org_x = x * math.cos(-theta) - y * math.sin(-theta)
                org_y = x * math.sin(-theta) + y * math.cos(-theta)
                
                if method == 'directional':
                    if derivation_order == 1:
                        kernel[i][j] = interp_func_x(org_x) * interp_func_y(abs(org_y))
                    elif derivation_order == 2:
                        kernel[i][j] = interp(r)
                elif method == "rotational":
                    
                    if derivation_order == 1:
                        if r == 0.0:
                            weight = 0.0
                        else:
                            weight = abs(org_x / r)
                        kernel[i][j] = interp(r) * weight 
                        if org_x < 0:
                            kernel[i][j] *= -1.0
                    elif derivation_order == 2:
                        kernel[i][j] = interp(r)

        if derivation_order == 1:
            kernel /= abs(kernel).max() 


        if derivation_order == 2:
            t = kernel.sum()/(kernel_size * kernel_size)
            kernel -= t
            kernel /= abs(kernel).max()

        ret = (xv, yv, kernel)
        
        return ret

