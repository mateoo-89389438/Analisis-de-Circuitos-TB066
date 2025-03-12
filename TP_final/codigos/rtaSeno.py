import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

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

figsize_x = 18
figsize_y = 8
fontsize_axis = 20
fontsize_legend = 20

archivos = ["rtaSeno1.txt", "rtaSeno2.txt", "rtaSeno3.txt"]
frecuencias = [5, 50, 2500]  

for i, archivo in enumerate(archivos):
    tiempo_sim, respuesta_sim = cargar_datos_ltspice(archivo)

    w = frecuencias[i]
    seno_s = w / (s**2 + w**2)

    x1_t = sp.inverse_laplace_transform(H1_s * seno_s, s, t)
    x1_t_func = sp.lambdify(t, x1_t, modules='numpy')
    x1_values = x1_t_func(t_values)
    x2_t = sp.inverse_laplace_transform(H2_s * seno_s, s, t)
    x2_t_func = sp.lambdify(t, x2_t, modules='numpy')
    x2_values = x2_t_func(t_values)
    seno_input = np.sin(w * t_values)

    plt.figure(figsize=(figsize_x, figsize_y))

    if w == frecuencias[0]:
        plt.plot(t_values, x1_values * 1000, label=f'Original', color="cornflowerblue", linestyle='-')
        plt.plot(t_values, x2_values * 1000, label=f'Normalizada', color="darkblue", linestyle='-')
        plt.plot(tiempo_sim, respuesta_sim * 1000, label=f'Simulada', color="crimson", linestyle='-')
        plt.plot(t_values, seno_input * 1000, label=f'Entrada', color='grey', linestyle='-')
        plt.xlabel('Tiempo [s]', fontsize=fontsize_axis)
        plt.ylabel('Tensión [mV]', fontsize=fontsize_axis)
    elif w == frecuencias[1]:
        plt.plot(t_values * 1000, x1_values, label=f'Original', color="cornflowerblue", linestyle='-')
        plt.plot(t_values * 1000, x2_values, label=f'Normalizada', color="darkblue", linestyle='-')
        plt.plot(tiempo_sim * 1000, respuesta_sim, label=f'Simulada', color="crimson", linestyle='-')
        plt.plot(t_values * 1000, seno_input, label=f'Entrada', color='grey', linestyle='-')
        plt.xlabel('Tiempo [ms]', fontsize=fontsize_axis)
        plt.ylabel('Tensión [V]', fontsize=fontsize_axis)
    elif w == frecuencias[2]:
        plt.plot(t_values * 1000, x1_values * 1000, label=f'Original', color="cornflowerblue", linestyle='-')
        plt.plot(t_values * 1000, x2_values * 1000, label=f'Normalizada', color="darkblue", linestyle='-')
        plt.plot(tiempo_sim * 1000, respuesta_sim * 1000, label=f'Simulada', color="crimson", linestyle='-')
        plt.plot(t_values * 1000, seno_input * 1000, label=f'Entrada', color='grey', linestyle='-')
        plt.xlabel('Tiempo [ms]', fontsize=fontsize_axis)
        plt.ylabel('Tensión [mV]', fontsize=fontsize_axis)

    plt.xlim(0, 3 * (2 * np.pi / w) * (1000 if w in [frecuencias[1], frecuencias[2]] else 1)) 
    plt.grid()
    plt.legend(fontsize=fontsize_legend, loc='lower left')
    plt.show()










