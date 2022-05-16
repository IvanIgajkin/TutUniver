import numpy as np
import math
import numdifftools as nd

def min(f, x0):
    ZERO = 1e-4 #"условный ноль" для использования при сравнении вещественных чисел  с нулем

    eps1, eps2 = 0.1, 0.15 #задаваемая точность
    M = 10 #предельное число итераций

    m0 = 100

    k = 0
    xk = x0 #задаём xk, как начальное приближение
    mk = m0

    # в python нет цикла вида do ... while, поэтому используем обычный while
    # и, так как новая точка не известна до начала работы цикла, проверку на сходимость нельзя поставить, как главное условие выхода из цикла
    while k < M:
        grad_value = nd.Gradient(f)(xk) #значение градиента для xk
        if np.linalg.norm(grad_value) <= eps1: #проверка на норму градиента функции для xk
            break

        Hesse_value = nd.Hessian(f)(xk) #матрица Гессе для xk
        dk = -np.linalg.inv(Hesse_value + mk * np.eye(len(Hesse_value))) @ grad_value
    
        xk_new = xk + dk
        mk = mk / 2.0 if f(xk_new) < f(xk) else 2.0 * mk
        
        xk = xk_new
        k = k + 1
    #цикл начинается сначала

    # заменить все значения, примерно равные нулю на 0.0
    return np.array(list(map(lambda xi: 0.0 if xi < ZERO else xi, xk)))

#задаём функцию
#f = lambda x: x[0]**2 - 4 * x[0]
#f = lambda x_vec: x
f = lambda x: (x[0] + 1)**3 / 3.0 + x[1]

#g = [lambda x: x[0] - 1]
g = [lambda x: 1 - x[0], lambda x: -x[1]]

def p (x, r):
    _sum = 0
    
    for gi in g:
        if gi(x):
            re
        _sum = _sum + 1 / gi(x)
        #_sum = _sum + math.log(-gi(x))
    
    res = -r * _sum
    return res

ZERO = 1e-1 #"условный ноль" для использования при сравнении вещественных чисел  с нулем

#начальное приближение
x0 = np.array([3., 2.5])
#x0 = np.array([8.0])
eps = 1.e-2 #задаваемая точность
M = 10 #предельное число итераций
r0 = 1

k = 0
xk = x0 #задаём xk, как начальное приближение
rk = r0

C = 4

while k < M:
    print('Iteration: {0}\tCurrent x = ({1})'.format(k + 1, xk))
    
    F = lambda x: f(x) + p(x, rk)
    xk = min(F, xk)

    if abs(p(xk, rk)) < eps: #проверка на сходимость
        break
    else:
        rk = rk / C
        k = k + 1
  #цикл начинается сначала

# заменить все значения, примерно равные нулю на 0.0
xk = list(map(lambda xi: 0.0 if xi < ZERO else xi, xk))

print()
print('Found local minimum on iterarion {0}'.format(k + 1))
print('Local minimum for the function is {0}'.format(xk))
