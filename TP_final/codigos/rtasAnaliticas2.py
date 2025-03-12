import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square  

figsize_x = 18
figsize_y = 8
fontsize_axis = 19
fontsize_legend = 19
color_original = 'darkblue'
color_normalizado = 'g'

s, t = sp.symbols('s t', real=True)


# H(s) original
k1 = 1005.96  
numerador1 = s**2  
denominador1 = (s + 800) * (s**2 + 25*s + 2500)  
H1_s = k1 * (numerador1 / denominador1)

# H(s) normalizada
k2 = 999.996  
numerador2 = s**2  
denominador2 = (s + 833) * (s**2 + 26*s + 2460)  
H2_s = k2 * (numerador2 / denominador2)


### respuesta al impulso ###
h1_t = sp.inverse_laplace_transform(H1_s, s, t)
h1_t_func = sp.lambdify(t, h1_t, modules='numpy')
h2_t = sp.inverse_laplace_transform(H2_s, s, t)
h2_t_func = sp.lambdify(t, h2_t, modules='numpy')
t_values = np.linspace(0, 5, 1000000)  
h1_values = h1_t_func(t_values)
h2_values = h2_t_func(t_values)

plt.figure(figsize=(figsize_x, figsize_y))
plt.plot(t_values * 1000, h1_values, label='Original', color=color_original)
plt.plot(t_values * 1000, h2_values, label='Normalizada', color=color_normalizado, linestyle='-')
plt.xlabel('Tiempo [ms]', fontsize=fontsize_axis)  
plt.ylabel('Tensión [V]', fontsize=fontsize_axis)
plt.xlim(0, 0.45 * 1000)  
plt.ylim(-60, 60)
plt.grid()
plt.legend(fontsize=fontsize_legend)
plt.show()


### respuesta al escalón ###
U_s = 1/s  

v1_t = sp.inverse_laplace_transform(H1_s * U_s, s, t)
v1_t_func = sp.lambdify(t, v1_t, modules='numpy')
v2_t = sp.inverse_laplace_transform(H2_s * U_s, s, t)
v2_t_func = sp.lambdify(t, v2_t, modules='numpy')
v1_values = v1_t_func(t_values)
v2_values = v2_t_func(t_values)

plt.figure(figsize=(figsize_x, figsize_y))
plt.plot(t_values * 1000, v1_values * 1000, label='Original', color=color_original)
plt.plot(t_values * 1000, v2_values * 1000, label='Normalizada', color=color_normalizado, linestyle='-')
plt.xlabel('Tiempo [ms]', fontsize=fontsize_axis)  
plt.ylabel('Tensión [mV]', fontsize=fontsize_axis)  
plt.xlim(0, 0.45 * 1000)  
plt.grid()
plt.legend(fontsize=fontsize_legend)
plt.show()



### respuestas al seno ###
frecuencias = [5, 50, 2500]  # en rad/s

for w in frecuencias:
    seno_s = w / (s**2 + w**2)
    x1_t_func = sp.lambdify(t, sp.inverse_laplace_transform(H1_s * seno_s, s, t), modules='numpy')
    x2_t_func = sp.lambdify(t, sp.inverse_laplace_transform(H2_s * seno_s, s, t), modules='numpy')
    seno_input = np.sin(w * t_values)
    
    plt.figure(figsize=(figsize_x, figsize_y))
    
    if w == frecuencias[0]:
        plt.plot(t_values, x1_t_func(t_values) * 1000, label=f'Original', color=color_original)
        plt.plot(t_values, x2_t_func(t_values) * 1000, label=f'Normalizada', color=color_normalizado, linestyle='-')
        plt.plot(t_values, seno_input * 1000, label=f'Entrada', color='grey', linestyle='-')
        plt.ylabel('Tensión [mV]', fontsize=fontsize_axis)  
        plt.xlabel('Tiempo [s]', fontsize=fontsize_axis) 
    elif w == frecuencias[1]:
        plt.plot(t_values * 1000, x1_t_func(t_values), label=f'Original', color=color_original)
        plt.plot(t_values * 1000, x2_t_func(t_values), label=f'Normalizada', color=color_normalizado, linestyle='-')
        plt.plot(t_values * 1000, seno_input, label=f'Entrada', color='grey', linestyle='-')
        plt.xlabel('Tiempo [ms]', fontsize=fontsize_axis)  
        plt.ylabel('Tensión [V]', fontsize=fontsize_axis)  
    elif w == frecuencias[2]:
        plt.plot(t_values * 1000, x1_t_func(t_values) * 1000, label=f'Original', color=color_original)
        plt.plot(t_values * 1000, x2_t_func(t_values) * 1000, label=f'Normalizada', color=color_normalizado, linestyle='-')
        plt.plot(t_values * 1000, seno_input * 1000, label=f'Entrada', color='grey', linestyle='-')
        plt.xlabel('Tiempo [ms]', fontsize=fontsize_axis)  
        plt.ylabel('Tensión [mV]', fontsize=fontsize_axis)  
    
    plt.xlim(0, 3 * (2 * np.pi / w) * (1000 if w in [frecuencias[1], frecuencias[2]] else 1))  
    plt.grid()
    plt.legend(fontsize=fontsize_legend)
    plt.show()



### respuestas a la cuadrada ###
for w in frecuencias:
    N = 9
    cuadrada_s = sum((4 / (sp.pi * (2 * n - 1))) * ((2 * n - 1) * w) / (s**2 + (2 * n - 1)**2 * w**2) for n in range(1, N))
    xq1_t_func = sp.lambdify(t, sp.inverse_laplace_transform(H1_s * cuadrada_s, s, t), modules='numpy')
    xq2_t_func = sp.lambdify(t, sp.inverse_laplace_transform(H2_s * cuadrada_s, s, t), modules='numpy')
    cuadrada_input = square(w * t_values, duty=0.5)
    
    plt.figure(figsize=(figsize_x, figsize_y))
    
    if w == frecuencias[0]:
        plt.plot(t_values, xq1_t_func(t_values) * 1000, label=f'Original', color=color_original)
        plt.plot(t_values, xq2_t_func(t_values) * 1000, label=f'Normalizada', color=color_normalizado, linestyle='-')
        plt.plot(t_values, cuadrada_input * 1000, label=f'Entrada', color='grey', linestyle='-')
        plt.ylabel('Tensión [mV]', fontsize=fontsize_axis)  
        plt.xlabel('Tiempo [s]', fontsize=fontsize_axis) 
    elif w == frecuencias[1]:
        plt.plot(t_values * 1000, xq1_t_func(t_values), label=f'Original', color=color_original)
        plt.plot(t_values * 1000, xq2_t_func(t_values), label=f'Normalizada', color=color_normalizado, linestyle='-')
        plt.plot(t_values * 1000, cuadrada_input, label=f'Entrada', color='grey', linestyle='-')
        plt.xlabel('Tiempo [ms]', fontsize=fontsize_axis)  
        plt.ylabel('Tensión [V]', fontsize=fontsize_axis)  
    elif w == frecuencias[2]:
        plt.plot(t_values * 1000, xq1_t_func(t_values) * 1000, label=f'Original', color=color_original)
        plt.plot(t_values * 1000, xq2_t_func(t_values) * 1000, label=f'Normalizada', color=color_normalizado, linestyle='-')
        plt.plot(t_values * 1000, cuadrada_input * 1000, label=f'Entrada', color='grey', linestyle='-')
        plt.xlabel('Tiempo [ms]', fontsize=fontsize_axis)  
        plt.ylabel('Tensión [mV]', fontsize=fontsize_axis)  
    
    plt.xlim(0, 3 * (2 * np.pi / w) * (1000 if w in [frecuencias[1], frecuencias[2]] else 1)) 
    plt.grid()
    plt.legend(fontsize=fontsize_legend)
    plt.show()





