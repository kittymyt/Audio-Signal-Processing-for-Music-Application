# approximating a fragment of a sound with this stochastic model
import numpy as np
from scipy.signal import get_window, resample
import math
import sys, os, time
from scipy.fftpack import fft, ifft
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models'))
import utilFunctions as UF

(fs, x) = UF.wavread("../../sounds/ocean.wav")
M = N = 256
stocf = 0.2
w = get_window('hanning', M)
xw = x[10000:10000+M] * w
X = fft(xw)
mX = 20 * np.log10(abs(X[:N//2]))
mXenv = resample(np.maximum(-200, mX), int((N//2+1)*stocf))  #approximation on DB spectrum