# read_open_bci.py

import struct as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scisig
import f_SignalProcFuncLibs as sigfun

def f_ReadTxtOpenBci(str_FileName, v_ChanNums):
    s_File = open(f'data/{str_FileName}', 'r')
    m_Data = np.zeros((1, len(v_ChanNums)))
    v_ChanNums = np.sort(v_ChanNums)
    while 1:
        str_Line = s_File.readline()
        if str_Line == '':
            break
        if "Sample Rate" in str_Line:
            str_Line = str_Line.split('=')
            s_FsHz = int(str_Line[1][:-3])
            continue
        str_Line = str_Line.split(',')
        if len(str_Line) < 2:
            continue
        if "Sample Index" in str_Line[0] or float(str_Line[0]) == 0.0:
            continue
        for s_ChannCount in range(len(v_ChanNums)):
            m_Data[-1, s_ChannCount] = float(str_Line[v_ChanNums[s_ChannCount] + 1])

        m_Data.resize((np.size(m_Data, 0) + 1, np.size(m_Data, 1)), refcheck=False)
        # if len(str_Line) < 40 or str_Line[0:11].lower() == 'sleep stage':
        #     continue
        # str_Line = str_Line.split(',')
        # v_SS = str_Line[3].split('-')

        # v_HypDur.append(int(str_Line[4]))
        # v_HypAbsTime.append(str_Line[2])

    m_Data.resize((np.size(m_Data, 0) - 1, np.size(m_Data, 1)), refcheck=False)
    return m_Data, s_FsHz

v_ChanNums = [7]
#str_FileName = './data/open_bci_temp_plant_test.txt'
#str_FileName = './data/OpenBci/OpenBCI-RAW-2022-02-22_10-13-40.txt'
str_FileName = 'noche_17_May.txt'
m_Data, s_FsHz = f_ReadTxtOpenBci(str_FileName, v_ChanNums)

noche = str_FileName[:6]


s_RefChann = 0
#v_TukeyWin = scisig.windows.tukey(np.size(m_Data, 0), 0.1)
#plt.plot(v_TukeyWin)
#v_SigRef = (m_Data[:,s_RefChann] - np.mean(m_Data[:,s_RefChann])) * v_TukeyWin
v_SigRef = m_Data[:,s_RefChann]
s_CutTime = 0
if s_CutTime:
    s_FirstSec = 1800
    s_LastSec = 2100
    s_FirstSam = int(s_FirstSec * s_FsHz)
    s_LastSam = int(s_LastSec * s_FsHz)
    v_SigRef = v_SigRef[s_FirstSam:s_LastSam]

v_TukeyWin = scisig.windows.tukey(len(v_SigRef), 0.01)
v_SigRef = (v_SigRef - np.mean(v_SigRef)) * v_TukeyWin
v_SigFilt = v_SigRef
v_TimeSig = np.arange(0, len(v_SigRef)) / s_FsHz
# sos = scisig.butter(5, 5, 'hp', fs=250, output='sos')
# v_SigFilt = scisig.sosfilt(sos, v_SigFilt)

plt.plot(v_TimeSig, v_SigRef)
plt.show()

#st_Filt = sigfun.f_GetIIRFilter(s_FsHz, [58.0, 62.0], [59.0, 61.0], p_Type='bs')
#v_SigFilt = sigfun.f_IIRBiFilter(st_Filt, m_Data[:,0])
# st_Filt = sigfun.f_GetIIRFilter(s_FsHz, 1.0, 5, p_Type='hp')
#st_Filt = sigfun.f_GetIIRFilter(s_FsHz, 50, 55, p_Type='lp')
st_Filt = sigfun.f_GetIIRFilter(s_FsHz, [5, 50], [0.1,59])
v_SigFilt = sigfun.f_IIRBiFilter(st_Filt, v_SigRef)

#v_TimeArray = np.arange(0,np.size(m_Data, 0)) / s_FsHz
s_SatPos = 150
s_SatNeg = -150
#v_SigFilt[v_SigFilt > s_SatPos] = s_SatPos
#v_SigFilt[v_SigFilt < s_SatNeg] = s_SatNeg

#plt.plot(m_Data[:,s_RefChann])
plt.plot(v_SigFilt)
plt.show()



# #     ----- TRANSFORMADA FOURIER ------
# v_FreqFFT = np.arange(0, np.size(v_SigRef)) * s_FsHz / np.size(v_SigRef)
# v_FFTSig = np.fft.fft(v_SigRef) # transformada de Fourier
# s_HalfInd = int(np.size(v_FFTSig) / 2)
# v_FFTSig = v_FFTSig[0:s_HalfInd]
# v_FreqFFT = v_FreqFFT[0:s_HalfInd]

# # plottear FFT (se puede escoger escala logaritmica o normal)
# plt.figure()
# # plt.plot(np.log10(v_FreqFFT), np.log10(np.abs(v_FFTSig)), linewidth=1)
# plt.plot(v_FreqFFT, np.abs(v_FFTSig), linewidth=1)
# plt.xlabel('Freq. (Hz)')
# plt.title("Transformada Fourier")
# plt.grid()
# # plt.savefig(f'Resultados/{str_FileName}/FFT_No60_{str_FileName}_canal{s_RefChann}.png')
# plt.show()



s_F1Hz = 2.5
s_F2Hz = 20
s_FreqResHz = 0.1
s_NumCycles = 5
m_ConvMat, v_TimeArray, v_FreqTestHz = sigfun.f_GaborTFTransform(v_SigRef,
                                                                 s_FsHz, s_F1Hz, s_F2Hz,
                                                                 s_FreqResHz, s_NumCycles)#,
                                                                 #p_TimeAveSec=0.0)

##
axhdl = plt.subplots(2, 1, sharex=True, constrained_layout=True)

# Graficamos la señal x1
# axhdl[1][0].plot(v_TimeArray, m_Data[:,s_RefChann], linewidth=1)
axhdl[1][0].plot(v_TimeArray, v_SigFilt, linewidth=1)
axhdl[1][0].set_ylabel("señal", fontsize=15)
axhdl[1][0].grid(1)

# Graficamos la matriz resultante en escala de colores
# ConvMatPlot = ConvMat
m_ConvMatPlot = np.abs(m_ConvMat)
immat = axhdl[1][1].imshow(m_ConvMatPlot, interpolation='none',
                           origin='lower', aspect='auto',
                           extent=[v_TimeArray[0], v_TimeArray[-1],
                                   v_FreqTestHz[0], v_FreqTestHz[-1]])

#immat.set_clim(np.min(m_ConvMatPlot), np.max(m_ConvMatPlot)*0.1)
axhdl[1][1].set_xlabel("Time (sec)", fontsize=15)
axhdl[1][1].set_ylabel("Freq (Hz)", fontsize=15)
axhdl[1][1].set_xlim([v_TimeArray[0], v_TimeArray[-1]])
immat.set_clim(np.min(m_ConvMatPlot), np.max(m_ConvMatPlot)*0.1)
axhdl[0].colorbar(immat, ax=axhdl[1][1])


plt.savefig(f'Resultados/{str_FileName[:-4]}/TiempoFrecuencia_ch{v_ChanNums[0]+1}_{int(s_F1Hz)}Hz_{s_F2Hz}Hz_{s_FirstSec}_{s_LastSec}')
plt.show()