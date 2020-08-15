import numpy as np
from scipy.signal import get_window
from scipy.fftpack import fft
import math
import matplotlib.pyplot as plt

### get a window

M = 63
window = get_window('blackmanharris', M)
hM1 = int(math.floor((M+1)/2))
hM2 = int(math.floor(M/2))

N = 512
hN = int(N/2)
fftbuffer = np.zeros(N)
fftbuffer[:hM1] = window[hM2:]
fftbuffer[N-hM2:] = window[:hM2]

### get the spectrum of the window

X = fft(fftbuffer)
absX = abs(X)
absX[absX<np.finfo(float).eps] = np.finfo(float).eps  #eps is the minimum value we can have in python
mX = 20 * np.log10(absX)
pX = np.angle(X)

mX1 = np.zeros(N)
pX1 = np.zeros(N)
mX1[:hN] = mX[hN:]
mX1[N-hN:] = mX[:hN]
pX1[:hN] = pX[hN:]
pX1[N-hN:] = pX[:hN]

# plot against X axis which has been normalized so the zero padding does not affect
# normalize the magnitude so that the maximum is zero decibels
plt.plot(np.arange(-hN, hN)/float(N)*M, mX1-max(mX1))
plt.axis([-20, 20, -100, 0])
plt.show()