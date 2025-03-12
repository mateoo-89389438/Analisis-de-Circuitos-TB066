import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

tiempo_sim, respuesta_sim = [], []

with open("rtaEscalon.txt", "r", encoding="latin-1") as file:
    for line in file.readlines()[1:]:  
        data = line.strip().split()
        tiempo_sim.append(float(data[0]))  
        respuesta_sim.append(float(data[1]))  


tiempo_sim = np.array(tiempo_sim)
respuesta_sim = np.array(respuesta_sim)

s, t = sp.symbols('s t', real=True)

# H1(s) original
k1 = 1005.96 
numerador1 = s**2  
denominador1 = (s + 800) * (s**2 + 25*s + 2500)  
H1_s = k1 * (numerador1 / denominador1)

# H2(s) normalizada
k2 = 999.996  
numerador2 = s**2  
denominador2 = (s + 833) * (s**2 + 26*s + 2460)  
H2_s = k2 * (numerador2 / denominador2)


figsize_x = 18
figsize_y = 8
fontsize_axis = 20
fontsize_legend = 22


U_s = 1 / s  
v1_t = sp.inverse_laplace_transform(H1_s * U_s, s, t)
v1_t_func = sp.lambdify(t, v1_t, modules='numpy')

v2_t = sp.inverse_laplace_transform(H2_s * U_s, s, t)
v2_t_func = sp.lambdify(t, v2_t, modules='numpy')

t_values = np.linspace(0, 0.45, 10000)

v1_values = v1_t_func(t_values)  
v2_values = v2_t_func(t_values)  


plt.figure(figsize=(figsize_x, figsize_y))
plt.plot(t_values * 1000, v1_values * 1000, label="Original", color="cornflowerblue", linestyle="-")
plt.plot(t_values * 1000, v2_values * 1000, label="Normalizada", color="darkblue", linestyle="-")
plt.plot(tiempo_sim * 1000, respuesta_sim * 1000, label="Simulada", color="crimson", linestyle="-")
plt.xlabel("Tiempo [ms]", fontsize=fontsize_axis)
plt.ylabel("Tension [mV]", fontsize=fontsize_axis)
plt.xlim(0, 0.40 * 1000)
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend(fontsize=fontsize_legend)  
plt.show()








