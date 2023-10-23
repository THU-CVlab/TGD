import  numpy as np
# Interpolation Class
class Interpolation(object):
    def __init__(self, x_lst, y_lst):
        self.x_lst = x_lst[::]
        self.y_lst = y_lst[::]

    # Find the interval [l, l+1] corresponding to x and estimate the value of that x
    #  (using the idea of equalization y = ((x - x0) * y0 + (x1 - x) * y1) / (x1 - x0) forward difference plus backward difference)
    def __call__(self, x):
        l = 0
        r = len(self.x_lst)
        while(l < r):
            m = (l + r) // 2                # floor
            if (self.x_lst[m] <= x):
                l = m + 1
            else:
                r = m
        l -= 1
        if l + 1 >= len(self.x_lst):
            return self.y_lst[-1]
        ret = ((x - self.x_lst[l]) * self.y_lst[l + 1] + (self.x_lst[l + 1] - x) * self.y_lst[l]) / (self.x_lst[l + 1] - self.x_lst[l])
        return ret
    
    # Find the up to frequency (midpoint of the interval), the window is small, just iterate through it
    def get_cutoff_frequency(self, y):
        start_index = int((len(self.y_lst) - 1) / 2)
        for index in range(start_index, len(self.y_lst) - 1):
            if self.y_lst[index] > y and self.y_lst[index + 1] < y:
                return (index + index + 1) / 2 - start_index
            elif self.y_lst[index] <= y:
                return index - start_index
        return len(self.y_lst) - start_index


# discrete Riemann integral
def integral(data, step):
    t = np.array(data).copy()
    for i in range(1, len(t)):
        t[i] = t[i] * step + t[i - 1]    
    return t


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import math
    x = [i for i in range(30)]
    y = [math.sin(i / 7) for i in range(30)]
    ip = Interpolation(x, y)
    plt.clf()
    plt.plot(x, y, 'x')
    xx = [i / 70 for i in range(1900)]
    yy = [ip(i) for i in xx]
    plt.plot(xx, yy)
    plt.show()
    plt.savefig('a.png')
