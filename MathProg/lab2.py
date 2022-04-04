import numpy as np
from scipy.optimize import fmin

#задаём функцию
#f = lambda x_vec: 2 * x_vec[0]**2 + x_vec[0] * x_vec[1] + x_vec[1]**2
f = lambda x_vec: 4 * (x_vec[0] - 5)**2 + (x_vec[1] - 6)**2

ZERO = 1e-1 #"условный ноль" для использования при сравнении вещественных чисел  с нулем

dim = 2
d_array = np.eye(dim) #массив, содержащий направления
d0 = d_array[dim - 1]

#метод для нахождения tk
def find_tk(yi, di):
    fi = lambda t: f(yi + t * di)
    t0 = 0.0

    return fmin(fi, t0, disp=False)[0]

def get_di(i):
    if i == 0:
        return d0
    elif i == dim - 1:
        return d_array[0]
    elif i == dim:
        return d_array[dim - 1]
    else:
        return d_array[i]

def set_yi_in_y_array(yi, i, k):
    if i >= len(y) or k == 0:
        y.append(yi)
    else:
        y[i] = yi

#начальное приближение
#x0 = np.array([0.5, 1.0])
x0 = np.array([8.0, 9.0])
eps1, eps2 = 0.1, 0.15 #задаваемая точность
M = 10 #предельное число итераций

k = 0
xk = x0 #задаём xk, как начальное приближение
y = [x0]

while k < M:
    print('Iteration: {0}\tCurrent x = (x1={1}, x2={2})'.format(k + 1, xk[0], xk[1]))
    
    is_founded = False
    i = 0
    
    while i <= dim:
        di = get_di(i)
        ti = find_tk(y[i], di)
        set_yi_in_y_array(y[i] + ti * di, i, k) #находим y[i + 1]
        if i == dim - 2: #i = n - 1
            if np.all(y[len(y) - 1] == y[0]):
                xk = y[len(y) - 1]
                is_founded = True
                break
            else:
                i = i + 1
        elif i == dim - 1: # i = n
            if np.all(y[dim] == y[1]):
                xk = y[dim]
                is_founded = True
                break
            else:
                i = i + 1
        else:
            i = i + 1
    #конец внутреннего цикла

    xk_new = np.array(y[-1])
    if np.linalg.norm(xk_new - xk) < eps1: #проверка на сходимость
        break

    xk = xk_new

    #строим новые направления
    d_new_array = d_array
    tmp_diff = y[-1] - y[1]
    for i in range(dim):
        d_new_array[i, dim - 1] = tmp_diff[i]

    if np.linalg.matrix_rank(d_new_array) == dim: #проверка на линейную независимость
        d0 = d_new_array[dim - 1]
        d_array = d_new_array
    else:
        d0 = d_array[dim - 1]

    y[0] = xk

    k = k + 1
  #цикл начинается сначала

# заменить все значения, примерно равные нулю на 0.0
xk = list(map(lambda xi: 0.0 if xi < ZERO else xi, xk))

print()
print('Found local minimum on iterarion {0}'.format(k + 1))
print('Local minimum for the function is (x1={0}, x2={1})'.format(xk[0], xk[1]))
