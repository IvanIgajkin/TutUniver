import numpy as np
import math
import numdifftools as nd
from scipy.optimize import fmin

#задаём функцию
#f = lambda x: x[0]**2 - 4 * x[0]
#f = lambda x_vec: x
f = lambda x: (x[0] - 4)**2 + (x[1] - 5)**2

#g = [lambda x: x[0] - 1]
g = [lambda x: x[0] + x[1] - 1]

def calc_A(x):
    return np.asarray([nd.Gradient(gi)(x) for gi in g])

def calc_tau(x):
    return np.asarray([-gi(x) for gi in g])

#метод для нахождения tk
def find_tk(xk, dk):
    fi = lambda t: f(xk + t * dk)
    t0 = 0.0

    return fmin(fi, t0, disp=False)[0]

ZERO = 1e-0 #"условный ноль" для использования при сравнении вещественных чисел  с нулем

#начальное приближение
x0 = np.array([3., 2.5])
#x0 = np.array([8.0])
eps = 1.e-2 #задаваемая точность
M = 10 #предельное число итераций

k = 0
xk = x0 #задаём xk, как начальное приближение

while k < M:
    print('Iteration: {0}\tCurrent x = ({1})'.format(k + 1, xk))
    
    Ak = calc_A(xk)
    tau_k = calc_tau(xk)
    d2xk = Ak.T @ np.linalg.inv(Ak @ Ak.T) @ tau_k
    d2xk_norm = np.linalg.norm(d2xk)
    
    grad_value = nd.Gradient(f)(xk)
    diff = Ak.T @ np.linalg.inv(Ak @ Ak.T) @ Ak
    dxk = -(np.eye(len(diff)) - diff) @ grad_value
    dxk_norm = np.linalg.norm(xk)

    if dxk_norm <= eps and d2xk_norm <= eps: #проверка на сходимость
        break
    elif dxk_norm > eps and d2xk_norm <= eps:
        d2xk = 0
    elif dxk_norm <= eps and d2xk_norm > eps:
        dxk = 0

    tk = find_tk(xk, dxk)
    xk = xk + tk * dxk + d2xk
    k = k + 1
  #цикл начинается сначала

# заменить все значения, примерно равные нулю на 0.0
xk = list(map(lambda xi: 0.0 if abs(xi) < ZERO else xi, xk))

print()
print('Found local minimum on iterarion {0}'.format(k + 1))
print('Local minimum for the function is {0}'.format(xk))
