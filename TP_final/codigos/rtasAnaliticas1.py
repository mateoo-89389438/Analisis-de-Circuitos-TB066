import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

figsize_x = 18
figsize_y = 8
fontsize_axis = 19
fontsize_legend = 19
color_in = 'g'
color_out = 'darkblue'

s, t = sp.symbols('s t', real=True)

k = 1005.96  
numerador = s**2  
denominador = (s + 800) * (s**2 + 25*s + 2500) 
H_s = k * (numerador / denominador)


### respuesta al impulso ###
h_t = sp.inverse_laplace_transform(H_s, s, t)
h_t_func = sp.lambdify(t, h_t, modules='numpy')
t_values = np.linspace(0, 0.8, 1000)
h_values = h_t_func(t_values)

plt.figure(figsize=(figsize_x, figsize_y))
plt.plot(t_values * 1000, h_values, color=color_out)
plt.xlabel('Tiempo [ms]', fontsize=fontsize_axis)          
plt.ylabel('Tensión [V]', fontsize=fontsize_axis)           
plt.xlim(0, 0.45 * 1000)
plt.ylim(-60, 60)
plt.grid()
plt.show()



### respuesta al escalón ###
U_s = 1/s
producto = H_s * U_s
v_t = sp.inverse_laplace_transform(producto, s, t)
v_t_func = sp.lambdify(t, v_t, modules='numpy')
v_values = v_t_func(t_values) 
heaviside = np.heaviside(t_values, 1)

plt.figure(figsize=(figsize_x, figsize_y))
plt.plot(t_values * 1000, heaviside * 1000, label='Entrada', color=color_in, linestyle='-')
plt.plot(t_values * 1000, v_values * 1000, label='Salida', color=color_out)
plt.legend(fontsize=fontsize_legend)  
plt.xlabel('Tiempo [ms]', fontsize=fontsize_axis)
plt.ylabel('Tensión [mV]', fontsize=fontsize_axis)
plt.xlim(0, 0.45 * 1000)
plt.grid()
plt.show()



### respuesta al seno (en w = 50 rad/s) ###
w = 50
seno_s = w / (s**2 + w**2)
producto = H_s * seno_s
x1_t = sp.inverse_laplace_transform(producto, s, t)
x1_t_func = sp.lambdify(t, x1_t, modules='numpy')
t_values = np.linspace(0, 0.8, 10000) 
x1_values = x1_t_func(t_values)
seno_input = np.sin(w * t_values)

plt.figure(figsize=(figsize_x, figsize_y))
plt.plot(t_values * 1000, seno_input * 1000, label=f'Entrada', color=color_in, linestyle='-')
plt.plot(t_values * 1000, x1_values * 1000, label='Salida', color=color_out)
plt.legend(fontsize=fontsize_legend)  
plt.xlabel('Tiempo [ms]', fontsize=fontsize_axis)
plt.ylabel('Tensión [V]', fontsize=fontsize_axis)
plt.xlim(0, 4.5*(2*np.pi/w) * 1000) 
plt.grid()
plt.show()
