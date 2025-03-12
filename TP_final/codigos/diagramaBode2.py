import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.interpolate import interp1d

figsize_x = 18
figsize_y = 10
fontsize_axis = 20
fontsize_legend = 22

frecuencia_sim, mag_sim, fase_sim = [], [], []

with open("data_bode_polar.txt", "r", encoding="latin-1") as file:
    for line in file.readlines()[1:]:  
        data = line.strip().split()
        freq = float(data[0])  
        mag_dB = float(data[1].split("dB")[0][1:]) 
        fase_deg = float(data[1].split("°")[0].split(",")[-1])  
        
        frecuencia_sim.append(freq)
        mag_sim.append(mag_dB)
        fase_sim.append(fase_deg)

frecuencia_sim = np.array(frecuencia_sim) * 2 * np.pi  

# H(s) original
numerador_original = [1005.96, 0, 0]  # 1005.96 * s^2
denominador_original = [1, 825, 22500, 2000000]  # s^3 + 825s^2 + 22500s + 2000000
H_s_original = signal.TransferFunction(numerador_original, denominador_original)

# H(s) normalizada
numerador_normalizado = [999.996, 0, 0]  # 999.996 * s^2
denominador_normalizado = [1, 859.307, 24107.642, 2049726.7]  # s^3 + 859.307s^2 + 24107.642s + 2049726.7
H_s_normalizado = signal.TransferFunction(numerador_normalizado, denominador_normalizado)


w = np.logspace(-1, 5, 1000)  

_, mag_original, phase_original = signal.bode(H_s_original, w)
_, mag_normalizado, phase_normalizado = signal.bode(H_s_normalizado, w)

# interpolacion de los datos de la simulación para suavizar la línea
mag_interp = interp1d(frecuencia_sim, mag_sim, kind='cubic', fill_value="extrapolate")
fase_interp = interp1d(frecuencia_sim, fase_sim, kind='cubic', fill_value="extrapolate")

fig, axs = plt.subplots(2, 1, figsize=(figsize_x, figsize_y))

axs[0].semilogx(w, mag_original, label='Original', color='cornflowerblue', linestyle='-')
axs[0].semilogx(w, mag_normalizado, label='Normalizada', color='darkblue', linestyle='-')
axs[0].semilogx(frecuencia_sim, mag_interp(frecuencia_sim), label='Simulada', color='crimson', linestyle='-')
axs[0].set_ylabel('Magnitud [dB]', fontsize=fontsize_axis)
axs[0].axvline(50, color='black', linestyle='--', linewidth=1)
axs[0].axvline(800, color='black', linestyle='--', linewidth=1,)
axs[0].set_xlim(0.1, 10000)
axs[0].set_ylim(-120, 20)
axs[0].grid(which="both", linestyle="--", linewidth=0.5)
axs[0].legend(fontsize=fontsize_legend)

axs[1].semilogx(w, phase_original, label='Original', color='cornflowerblue', linestyle='-')
axs[1].semilogx(w, phase_normalizado, label='Normalizada', color='darkblue', linestyle='-')
axs[1].semilogx(frecuencia_sim, fase_interp(frecuencia_sim), label='Simulada', color='crimson', linestyle='-')
axs[1].set_xlabel('Frecuencia [rad/s]', fontsize=fontsize_axis)
axs[1].set_ylabel('Fase [deg]', fontsize=fontsize_axis)
axs[1].axvline(50, color='black', linestyle='--', linewidth=1)
axs[1].axvline(800, color='black', linestyle='--', linewidth=1)
axs[1].set_xlim(0.1, 10000)
axs[1].set_ylim(-100, 200)
axs[1].grid(which="both", linestyle="--", linewidth=0.5)
axs[1].legend(fontsize=fontsize_legend)

plt.tight_layout()
plt.show()
