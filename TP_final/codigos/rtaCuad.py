import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square

figsize_x = 18
figsize_y = 8
fontsize_axis = 20
fontsize_legend = 19

def cargar_datos_ltspice(nombre_archivo):
    tiempo, respuesta = [], []
    with open(nombre_archivo, "r", encoding="latin-1") as file:
        for line in file.readlines()[1:]:  
            data = line.strip().split()
            tiempo.append(float(data[0]))  
            respuesta.append(float(data[1]))  
    return np.array(tiempo), np.array(respuesta)

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

t_values = np.linspace(0, 5, 1000000)  

archivos = ["rtaCuad1.txt", "rtaCuad2.txt", "rtaCuad3.txt"]
frecuencias = [5, 50, 2500]  # en rad/s

for w, archivo in zip(frecuencias, archivos):
    tiempo_sim, respuesta_sim = cargar_datos_ltspice(archivo)
    N = 9
    cuadrada_s = sum((4 / (sp.pi * (2 * n - 1))) * ((2 * n - 1) * w) / (s**2 + (2 * n - 1)**2 * w**2) for n in range(1, N))
    xq1_t_func = sp.lambdify(t, sp.inverse_laplace_transform(H1_s * cuadrada_s, s, t), modules='numpy')
    xq2_t_func = sp.lambdify(t, sp.inverse_laplace_transform(H2_s * cuadrada_s, s, t), modules='numpy')
    xq1_t = np.squeeze(xq1_t_func(t_values))
    xq2_t = np.squeeze(xq2_t_func(t_values))
    cuadrada_input = square(w * t_values, duty=0.5)
    
    if w in [frecuencias[1], frecuencias[2]]:
        t_values_ms = t_values * 1000  
        tiempo_sim_ms = tiempo_sim * 1000  
        x_label = 'Tiempo [ms]'  
    else:
        t_values_ms = t_values  
        tiempo_sim_ms = tiempo_sim  
        x_label = 'Tiempo [s]'  
    
    plt.figure(figsize=(figsize_x, figsize_y))
    plt.plot(t_values_ms, xq1_t, label=f'Original', color='cornflowerblue')
    plt.plot(t_values_ms, xq2_t, label=f'Normalizada', color='darkblue', linestyle='-')
    plt.plot(tiempo_sim_ms, respuesta_sim, label=f'Simulada', color="crimson", linestyle='-')
    plt.plot(t_values_ms, cuadrada_input, label=f'Entrada', color='grey', linestyle='-')
    plt.xlabel(x_label, fontsize=fontsize_axis)  
    plt.ylabel('Tensión [V]', fontsize=fontsize_axis)
    plt.xlim(0, 3 * (2 * np.pi) / w * (1000 if w in [50, 2500] else 1))  
    plt.grid()
    plt.legend(fontsize=fontsize_legend)
    plt.show()










    