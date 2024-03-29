import numpy as np
import numdifftools as nd

#задаём функцию
#f = lambda x_vec: 2 * x_vec[0]**2 + x_vec[0] * x_vec[1] + x_vec[1]**2
f = lambda x_vec: 4 * (x_vec[0] - 4)**2 + (x_vec[1] - 3)**2

ZERO = 1e-4 #"условный ноль" для использования при сравнении вещественных чисел  с нулем

x0 = (0.5, 1) #начальное приближение
eps1, eps2 = 0.1, 0.15 #задаваемая точность
M = 10 #предельное число итераций

m0 = 20

k = 0
xk = x0 #задаём xk, как начальное приближение
mk = m0

# в python нет цикла вида do ... while, поэтому используем обычный while
# и, так как новая точка не известна до начала работы цикла, проверку на сходимость нельзя поставить, как главное условие выхода из цикла
while k < M:
    print('Iteration: {0}\tCurrent x = (x1={1}, x2={2})\tmu_k={3}'.format(k + 1, xk[0], xk[1], mk))

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
xk = list(map(lambda xi: 0.0 if xi < ZERO else xi, xk))

print()
print('Found local minimum on iterarion {0}'.format(k + 1))
print('Local minimum for the function is (x1={0}, x2={1})'.format(xk[0], xk[1]))
