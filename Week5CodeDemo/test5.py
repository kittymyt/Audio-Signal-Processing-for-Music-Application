# sinusoidal synthesis

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../software/models/'))
import dftModel as DFT
import utilFunctions as UF
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft

(fs,x) = UF.wavread('../../sounds/oboe-A4.wav')
Ns = 512
hNs = Ns/2
H = Ns/4
M = 511
t = -70
w = get_window('hamming', M)
x1 = x[int(.8*fs):int(.8*fs)+M]
mX,pX = DFT.dftAnal(x1, w, Ns)
ploc = UF.peakDetection(mX,t)
iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc) #interpolate the peaks to get more accurate value of the location of the sinusoid
ipfreq = fs*iploc/float(Ns)
Y = UF.genSpecSines_p(ipfreq, ipmag, ipphase, Ns, fs)  # do the synthesis
y = np.real(ifft(Y))   # inverse fft

# undo the window
sw = np.zeros(Ns)
ow = triang(Ns/2)
sw[int(hNs-H):int(hNs+H)] = ow
bh = blackmanharris(Ns)
bh = bh/sum(bh)
sw[int(hNs-H):int(hNs+H)] = sw[int(hNs-H):int(hNs+H)]/bh[int(hNs-H):int(hNs+H)]

yw = np.zeros(Ns)
yw[:int(hNs-1)] = y[int(hNs+1):]
yw[int(hNs-1):] = y[:int(hNs+1)]
yw *= sw

# plots the result magnitude spectrum
freqaxis = fs*np.arange(Ns/2+1)/float(Ns)
plt.plot(freqaxis, mX)
plt.plot(fs * iploc / Ns, ipmag, marker='x', linestyle='')

plt.show()