import numpy as np
import numdifftools as nd
from scipy.optimize import fmin

#задаём функцию
#f = lambda x_vec: 2 * x_vec[0]**2 + x_vec[0] * x_vec[1] + x_vec[1]**2
f = lambda x_vec: 4 * (x_vec[0] - 5)**2 + (x_vec[1] - 6)**2

ZERO = 1e-4 #"условный ноль" для использования при сравнении вещественных чисел  с нулем

dim = 2
d_array = np.eye(dim) #массив, содержащий направления
n = len(d_array) - 1
d0 = d_array[n]

#метод для нахождения tk
def find_tk(yi, di):
    fi = lambda t: f(yi + t * di)
    t0 = 0.0

    return fmin(fi, t0, disp=False)[0]

x0 = np.array([8.0, 9.0]) #начальное приближение
eps1, eps2 = 0.1, 0.15 #задаваемая точность
M = 10 #предельное число итераций

k = 0
xk = x0 #задаём xk, как начальное приближение
y = [x0]

while k < M:
    is_founded = False
    i = 0
    while i < dim:
        ti = find_tk(y[i], d_array[i])
        y.append(y[i] + ti * d_array[i]) #находим y[i + 1]
        if i == n - 1:
            if np.all(y[n] == y[0]):
                xk = y[n]
                is_founded = True
                break
            else:
                i = i + 1
        elif i == n:
            if np.all(y[n + 1] == y[1]):
                xk = y[n + 1]
                is_founded = True
                break
            else:
                i = i + 1
        else:
            i = i + 1


    print(y)
    xk_new = np.array(y[n + 1])
    if np.linalg.norm(xk_new - xk) < eps1: #проверка на сходимость
        break

    xk = xk_new

    for i in range(dim - 1):
        d_array[i] = d_array[i + 1]

    d_array[n] = y[n + 1] - y[1]
    d0 = d_array[n]

    y = [xk]

    k = k + 1
  #цикл начинается сначала

# заменить все значения, примерно равные нулю на 0.0
xk = list(map(lambda xi: 0.0 if xi < ZERO else xi, xk))

print(xk)
