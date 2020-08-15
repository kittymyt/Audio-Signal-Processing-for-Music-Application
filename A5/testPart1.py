import numpy as np
from scipy.signal import get_window
import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import dftModel as DFT
import utilFunctions as UF
import matplotlib.pyplot as plt

window = 'blackman'
t = -40
inputFile = "../../sounds/sine-200.wav"
f = 200.0

### Your code here
(fs,x) = UF.wavread(inputFile)

for k in range(1,20):
    M = 100*k + 1
    hM = int(math.floor(M/2))
    N = np.power(2,int(np.log2(M))+1)
    w = get_window(window, M)

    x1 = x[int(.5*fs)-hM-1:int(.5*fs)+hM]
    mX,pX = DFT.dftAnal(x1,w,N)
    ploc = UF.peakDetection(mX, t)
    (iploc, ipmag, ipphase) = UF.peakInterp(mX, pX, ploc)
    fEst = fs*iploc[0]/float(N)
    if (abs(f-fEst) < 0.05):
        result = (fEst, M, N)
        break