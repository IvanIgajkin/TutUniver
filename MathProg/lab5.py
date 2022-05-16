import numpy as np
import math
from scipy.optimize import fmin
from functools import reduce

#задаём функцию
f = lambda x: x[0]**2 - 4 * x[0]
#f = lambda x_vec: x

g = [lambda x: x[0] - 1]

def p (x, r):
    _sum = 0
    if isinstance(x, float):
        x = np.array([x])
    
    for gi in g:
        _sum = _sum + gi(x)
        #_sum = _sum + math.log(gi(x))
    
    res = r * _sum
    return res

ZERO = 1e-1 #"условный ноль" для использования при сравнении вещественных чисел  с нулем

#начальное приближение
#x0 = np.array([0.5, 1.0])
x0 = np.array([8.0])
eps = 0.1 #задаваемая точность
M = 10 #предельное число итераций
r0 = 1

k = 0
xk = x0 #задаём xk, как начальное приближение
rk = r0

C = 10

while k < M:
    if isinstance(xk, float):
        xk = np.array([xk])
    
    print('Iteration: {0}\tCurrent x = (x1={1})'.format(k + 1, xk[0]))
    
    F = lambda x: f(x) + p(x, rk)
    xk = fmin(F, xk, disp=False)[0]

    if abs(p(xk, rk)) < eps: #проверка на сходимость
        break
    else:
        rk = rk / C
        k = k + 1
  #цикл начинается сначала

if isinstance(xk, float):
        xk = np.array([xk])

# заменить все значения, примерно равные нулю на 0.0
xk = list(map(lambda xi: 0.0 if xi < ZERO else xi, xk))

print()
print('Found local minimum on iterarion {0}'.format(k + 1))
print('Local minimum for the function is (x1={0})'.format(xk[0]))