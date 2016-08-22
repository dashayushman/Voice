import numpy as np
import matplotlib.pyplot as plt

np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
t = np.arange(256)
sp = np.fft.fft(np.sin(t))
print(sp.shape)
print(sp.real.shape)
freq = np.fft.fftfreq(t.shape[-1])
print(freq)
'''
plt.plot(freq, sp.real, freq, sp.imag)
plt.show()
'''