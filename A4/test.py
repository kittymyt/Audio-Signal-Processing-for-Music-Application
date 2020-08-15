import os
import sys
import numpy as np
import math
from scipy.signal import get_window
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import stft as STFT
import utilFunctions as UF
eps = np.finfo(float).eps

inputFile = '../../sounds/sax-phrase-short.wav'
N = 1024
M = 512
H = 64
window = 'hamming'

# calculate the signal energy
if(M%2):
	M = M-1

(fs,x) = UF.wavread(inputFile)
Esignal = sum(np.power(x,2))

# do the stft convert
w = get_window(window, M, True)
y = STFT.stft(x, w, N, H) 
y2 = y[M:-M]

# calculate the output signal engery
noise = abs(x-y)
Enoise = sum(np.power(noise,2))
x2 = x[M:-M]
noise2 = abs(x2-y2)
Enoise2 = sum(np.power(noise2,2))

# calculate the SNR
SNR1 = 10 * np.log10(Esignal/Enoise)
SNR2 = 10 * np.log10(Esignal/Enoise2)