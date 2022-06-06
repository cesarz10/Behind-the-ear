import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from f_SignalProcFuncLibs import *

# style.use('ggplot')
filename = 'noche_20feb'

s_FsHz = 250 # sample frequency
data = pd.read_csv(f'{filename}.txt')
# print(data.head())

data_real = data.to_numpy() # df to matrix
m_Data = data_real[2:, :]
# print(m_Data[:10, 1])

s_RefChann = 1 # reference channel -> queda igual que en el openBCI
v_Time = np.arange(0, np.size(m_Data, 0)) / s_FsHz # arreglo para el tiempo del tamaño de m_Data 
# print(f'Length: {len(m_Data[:, s_RefChann])}')
# third_1 = int(len(m_Data[:, s_RefChann]) / 3)
# mychan = m_Data[:,s_RefChann]
# print(mychan[:10]) # primeros 10 datos del canal elegido

m_Data_60 = bandPassFilter(m_Data[:, s_RefChann], 55, 65)
# m_Data_60_015 = bandPassFilter(m_Data_60, 0.0001, 0.20)

# # Ploteando la señal cruda del canal s_RefChann
# plt.plot(v_Time, m_data_60_015)
# plt.title(f'Señal Cruda C{s_RefChann}')
# plt.xlabel('Tiempo (sec)')
# plt.ylabel('Amplitud')
# plt.grid()
# plt.savefig(f'Resultados/{filename}/Cruda_No60_020_{filename}_canal{s_RefChann}.png')
# plt.show()


# # ------------------------------------------------- FFT y algunos filtros ---------------------------------------

# FFT
v_Sig1 = m_Data_60 - np.mean(m_Data_60) # filtro promedio
v_FreqFFT = np.arange(0, np.size(v_Sig1)) * s_FsHz / np.size(v_Sig1)
v_FFTSig = np.fft.fft(v_Sig1) # transformada de Fourier
s_HalfInd = int(np.size(v_FFTSig) / 2)
v_FFTSig = v_FFTSig[0:s_HalfInd]
v_FreqFFT = v_FreqFFT[0:s_HalfInd]

# # plottear FFT (se puede escoger escala logaritmica o normal)
# plt.figure()
# # plt.plot(np.log10(v_FreqFFT), np.log10(np.abs(v_FFTSig)), linewidth=1)
# plt.plot(v_FreqFFT, np.abs(v_FFTSig), linewidth=1)
# plt.xlabel('Freq. (Hz)')
# plt.title("Transformada Fourier")
# plt.grid()
# plt.savefig(f'Resultados/{filename}/FFT_No60_{filename}_canal{s_RefChann}.png')
# plt.show()


# Filtrar señal y obtener la magnitud usando transformada de Hilbert
filt_FiltSOS = f_GetIIRFilter(s_FsHz, [6, 10], [5, 11]) # Infinite Impulse Response filter
v_SigFilt = f_IIRBiFilter(filt_FiltSOS, m_Data_60)
v_SigFilt1 = f_FFTFilter(m_Data_60, s_FsHz, [[6, 10]]) 
v_SigFiltAbs = np.abs(v_SigFilt) # valor absoluto de SigFilt
v_SigFiltMag = np.abs(signal.hilbert(v_SigFilt)) # Transformada de Hilbert


# # plottear resultados después de filtrar (se ven n segundos de señal: n*s_FsHz)
# n_sec = 14000

# plt.figure()
# # plt.plot(v_Time[:2*s_FsHz], m_Data[:2*s_FsHz,s_RefChann], alpha=0.3, linewidth=1, label="Cruda")
# # plt.plot(v_Time[2000000:2000000+n_sec*s_FsHz], v_SigFilt[2000000:2000000+n_sec*s_FsHz], color='#DB222A', alpha=0.7, linewidth=1, label="IIRBi")
# # plt.plot(v_Time[:n_sec*s_FsHz], v_SigFilt1[:n_sec*s_FsHz], 'limegreen', alpha=0.7, linewidth=1, label="FFT filter")
# plt.plot(v_Time[:n_sec*s_FsHz], v_SigFiltMag[:n_sec*s_FsHz], 'm', linewidth=2, label="Envolvente")
# plt.title("Envolvente")
# plt.xlabel("Tiempo (s)")
# plt.ylabel("Amplitud")
# plt.legend()
# plt.grid()
# plt.savefig(f'Resultados/{filename}/Envolvente_No60_015_{filename}_full_night_canal{s_RefChann}.png')
# plt.show()


# # # # --------------------------------------- FILTRANDO POR BANDAS -------------------------------

# # # # Filtrado en la banda Theta
# # # filt_FiltThetaSOS = f_GetIIRFilter(s_FsHz, [6, 10], [3, 13])
# # # v_SigFiltTheta = f_IIRBiFilter(filt_FiltThetaSOS, m_Data[:,s_RefChann])
# # # v_SigFiltThetaPha = np.angle(signal.hilbert(v_SigFiltTheta)) # fase de señal filtrando con theta
# # # v_SigFiltThetaMag = np.abs(signal.hilbert(v_SigFiltTheta)) # Magnitud de señal filtrando con theta


# # # # Filtrado en Gamma
# # # # Gamma slow
# # # filt_FiltGamma_slow = f_GetIIRFilter(s_FsHz, [30, 60], [25, 65])
# # # v_SigFiltGamma3_slow = f_IIRBiFilter(filt_FiltGamma_slow, m_Data[:,s_RefChann])
# # # v_SigFiltGamma3Mag_slow = np.abs(signal.hilbert(v_SigFiltGamma3_slow)) # Magnitud de señal gamma slow

# # # # Gamma medium
# # # filt_FiltGamma_medium = f_GetIIRFilter(s_FsHz, [60, 90], [55, 95])
# # # v_SigFiltGamma3_medium = f_IIRBiFilter(filt_FiltGamma_medium, m_Data[:,s_RefChann])
# # # v_SigFiltGamma3Mag_medium = np.abs(signal.hilbert(v_SigFiltGamma3_medium)) # Magnitud de señal gamma medium

# # # # Gamma fast
# # # filt_FiltGamma_fast = f_GetIIRFilter(s_FsHz, [90, 140], [85, 145])
# # # v_SigFiltGamma3_fast = f_IIRBiFilter(filt_FiltGamma_fast, m_Data[:,s_RefChann])
# # # v_SigFiltGamma3Mag_fast = np.abs(signal.hilbert(v_SigFiltGamma3_fast)) # Magnitud de señal gamma fast


# # # # subplot de las diferentes fecuencias (slow/medium/fast) -> se ve 1 segundo de señal
# # # figs, axs = plt.subplots(1, 3, figsize=(12, 8), sharey=False, constrained_layout=True)
# # # plt.suptitle('Magnitud de señales')

# # # axs[0].plot(v_Time[:n_sec*s_FsHz], v_SigFiltTheta[:n_sec*s_FsHz], color='#BFDBF7', linewidth=1)
# # # # axs[0].plot(v_Time[:n_sec*s_FsHz], v_SigFiltThetaMag[:n_sec*s_FsHz], color='#DB222A', linewidth=1)
# # # axs[0].set_xlabel('Tiempo (s)')
# # # axs[0].set_ylabel('Amplitud')
# # # axs[0].set_title('Theta')

# # # axs[1].plot(v_Time[:n_sec*s_FsHz], v_SigFiltGamma3_slow[:n_sec*s_FsHz], color='#BFDBF7', linewidth=1)
# # # # axs[1].plot(v_Time[:n_sec*s_FsHz], v_SigFiltGamma3Mag_slow[:n_sec*s_FsHz], color='#DB222A', linewidth=1)
# # # axs[1].set_xlabel('Tiempo (s)')
# # # axs[1].set_ylabel('Amplitud')
# # # axs[1].set_title('Gamma Slow')

# # # axs[2].plot(v_Time[:n_sec*s_FsHz], v_SigFiltGamma3_medium[:n_sec*s_FsHz], color='#BFDBF7', linewidth=1)
# # # # axs[2].plot(v_Time[:n_sec*s_FsHz], v_SigFiltGamma3Mag_medium[:n_sec*s_FsHz], color='#DB222A', linewidth=1)
# # # axs[2].set_ylabel('Amplitud')
# # # axs[2].set_xlabel('Tiempo (s)')
# # # axs[2].set_title('Gamma Medium')

# # # plt.savefig(f'Resultados/{filename}/Mag_Theta_Gamma_S_M_{filename}_Canal{s_RefChann}_{n_sec}s.png')
# # # plt.show()



# # ------------------------------------------ Transformada Tiempo-Frecuencia ----------------------------
ndatos = 5000



s_F1Hz = 0.1
s_F2Hz = 5
s_FreqResHz = 0.1
s_NumCycles = 5
m_ConvMat, v_TimeArray, v_FreqTestHz = \
    f_GaborTFTransform(m_Data_60[2000000:2000000+ndatos], s_FsHz,
                       s_F1Hz, s_F2Hz, s_FreqResHz, s_NumCycles)


# Graficamos la señal x1 con su correspondiente matriz de evaluación de patrones
axhdl = plt.subplots(2, 1, sharex=True, constrained_layout=True)

# Graficamos la señal x1
axhdl[1][0].plot(v_TimeArray, m_Data_60[2000000:2000000+ndatos], linewidth=1)
axhdl[1][0].set_ylabel("señal", fontsize=15)
axhdl[1][0].grid(1)

# Graficamos la matriz resultante en escala de colores
# ConvMatPlot = ConvMat
m_ConvMatPlot = np.abs(m_ConvMat)
immat = axhdl[1][1].imshow(m_ConvMatPlot, interpolation='none',
                           origin='lower', aspect='auto',
                           extent=[v_TimeArray[0], v_TimeArray[-1],
                                   v_FreqTestHz[0], v_FreqTestHz[-1]])

# immat.set_clim(np.min(m_ConvMatPlot), np.max(m_ConvMatPlot)*0.1)
axhdl[1][1].set_xlabel("Time (sec)", fontsize=15)
axhdl[1][1].set_ylabel("Freq (Hz)", fontsize=15)
axhdl[1][1].set_xlim([v_TimeArray[0], v_TimeArray[-1]])
axhdl[0].colorbar(immat, ax=axhdl[1][1])
plt.savefig(f'Resultados/{filename}/60_bndstp_{int(ndatos/1000)}Kdata_{filename}_Ch{s_RefChann}.png')
plt.show()  