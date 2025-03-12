import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

figsize_x = 18
figsize_y = 10
fontsize_axis = 20
fontsize_legend = 22

# H(s) original
numerador_original = [1005.96, 0, 0]  # 1005.96 * s^2
denominador_original = [1, 825, 22500, 2000000]  # (s + 800)(s^2 + 25s + 2500) = s^3 + 825s^2 + 22500s + 2000000
H_s_original = signal.TransferFunction(numerador_original, denominador_original)

# H(s) normalizada
numerador_normalizado = [999.996, 0, 0]  # 999.996 * s^2
denominador_normalizado = [1, 859.307, 24107.642, 2049726.7]  # (s + 833.333)(s^2 + 25.974s + 2459.662) = s^3 + 859.307s^2 + 24107.642s + 2049726.7
H_s_normalizado = signal.TransferFunction(numerador_normalizado, denominador_normalizado)

w, mag_original, phase_original = signal.bode(H_s_original)
_, mag_normalizado, phase_normalizado = signal.bode(H_s_normalizado, w)

frecuencias = [50, 800]  
frecuencias_rad = [2 * np.pi * f for f in frecuencias]  

fig, axs = plt.subplots(2, 1, figsize=(figsize_x, figsize_y))  

axs[0].semilogx(w, mag_original, label='Original', color='mediumblue', linestyle='-')
axs[0].semilogx(w, mag_normalizado, label='Normalizada', color='g', linestyle='-')
axs[0].set_ylabel('Magnitud [dB]', fontsize=fontsize_axis)  
axs[0].axvline(50, color='black', linestyle='--', linewidth=1)
axs[0].axvline(800, color='black', linestyle='--', linewidth=1,)
axs[0].set_xlim(0.1, 10000)
axs[0].set_ylim(-120, 20)
axs[0].grid(which="both", linestyle="-", linewidth=0.5)
axs[0].legend(fontsize=fontsize_legend)  

axs[1].semilogx(w, phase_original, label='Original', color='mediumblue', linestyle='-')
axs[1].semilogx(w, phase_normalizado, label='Normalizada', color='g', linestyle='-')
axs[1].set_xlabel('Frecuencia [rad/s]', fontsize=fontsize_axis)  
axs[1].set_ylabel('Fase [deg]', fontsize=fontsize_axis)          
axs[1].axvline(50, color='black', linestyle='--', linewidth=1)
axs[1].axvline(800, color='black', linestyle='--', linewidth=1)
axs[1].set_xlim(0.1, 10000)
axs[1].set_ylim(-100, 200)
axs[1].grid(which="both", linestyle="-", linewidth=0.5)
axs[1].legend(fontsize=fontsize_legend)  

plt.tight_layout()
plt.show()










