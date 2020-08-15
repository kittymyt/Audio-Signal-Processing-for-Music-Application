import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../software/models/"))
import dftModel as DFT
import utilFunctions as UF

(fs,x) = UF.wavread("../../sounds/sine-440.wav")
M = 501  # window size
N = 512*4  # fftsize
t = -20  # threshold for peak detection

w = get_window('hamming', M)
x1 = x[int(.8*fs):int(.8*fs)+M]    #get the middle of the sound
mX,pX = DFT.dftAnal(x1,w,N)
ploc = UF.peakDetection(mX, t)
pmag = mX[ploc]
iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)

freqaxis = fs*np.arange(N/2+1)/float(N)
plt.plot(freqaxis, mX)
plt.plot(fs * iploc / float(N), ipmag, marker='x', linestyle='')

plt.show()