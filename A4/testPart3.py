import os
import sys
import numpy as np
import math
from scipy.signal import get_window
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import stft
import utilFunctions as UF
eps = np.finfo(float).eps


inputFile = "../../sounds/piano.wav"
window = 'blackman'
M = 513
N = 1024
H = 128

fs,x = UF.wavread(inputFile)
w = get_window(window, M)

mX,pX = stft.stftAnal(x,w,N,H)
r,c = np.shape(mX)
mXLine = np.power(10,mX/20.0)

bin_freqs = np.arange(N) * fs / float(N)

temp1 = np.where(bin_freqs > 0)[0]
temp2 = np.where(bin_freqs < 3000)[0]
band_low = np.intersect1d(temp1,temp2)

temp3 = np.where(bin_freqs > 3000)[0]
temp4 = np.where(bin_freqs < 10000)[0]
band_high = np.intersect1d(temp3,temp4)

engEnv = np.zeros((r,2))
ODF = np.zeros((r,2))

# calculate energy per band
low_band_energy = np.sum(mXLine[:,band_low]**2, axis = 1)
low_band_energy = 10 * np.log10(low_band_energy)

engEnv[:,0] = low_band_energy

# calculate the high-band frequency
high_band_energy = np.sum(mXLine[:,band_high]**2, axis = 1)
high_band_energy = 10 * np.log10(high_band_energy)

engEnv[:,1] = high_band_energy

low_band_energy_roll = np.roll(low_band_energy, 1)
low_band_energy_roll[0] = 0
ODF_energy_low_band = low_band_energy - low_band_energy_roll
ODF[:,0] = [0 if i < 0 else i for i in ODF_energy_low_band]

high_band_energy_roll = np.roll(high_band_energy, 1)
high_band_energy_roll[0] = 0
ODF_energy_high_band = high_band_energy - high_band_energy_roll
ODF[:,1] = [0 if i < 0 else i for i in ODF_energy_high_band]
#ODF[ODF[:,1] < 0] = 0

#engEnv = np.append([low_band_energy],[high_band_energy], axis = 0)
#engEnv = np.transpose(engEnv)
