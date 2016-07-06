from sklearn import preprocessing
import math
import matplotlib.pyplot as plt
import numpy as np
from features import mfcc
from features import sigproc
from scipy import signal
import pywt

#concatnate methods for extracting features in a single loop
#can be done later but the present approach will cause a huge drop in performance

max_abs_scaler = preprocessing.MaxAbsScaler()

window_size = 20
overlap_size = 7

def meanScale(data):
    return preprocessing.scale(data)

def absScale(data):
    nor_data = max_abs_scaler.fit_transform(data)
    return nor_data

def absScale(data):
    return preprocessing.scale(data)

def getFeatures(data,window=False):
    #normalize the data
    #data = absScale(data)

    #extract MFCC features
    #mfcc_feat = mfcc(data,samplingRate)

    #Continuous wavelet transform
    #widths = np.arange(1, 100)
    #cwtmatr = signal.cwt(data, signal.ricker, widths)
    #cwtmatr = np.transpose(cwtmatr)

    #discrete wavelet transform
    #cA, cD = pywt.dwt(data, 'haar')
    #d_wavelet_features =  np.transpose(np.array([cA,cD]))

    '''
    print(wavelet_features)
    plt.imshow(wavelet_features, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(),vmin=-abs(cwtmatr).max())
    plt.show()
    '''

    #Create overlapping windows
    if window==False:
        ovlp_windows = [data]
    else:
        ovlp_windows = get_sliding_windows(data)
    #apply hamming window function
    windowed_frames = windowfn(ovlp_windows)

    # Time domain features
    window_gc,window_zc,window_len,window_rms,window_mean,window_var,window_ssi,window_iemg,window_peaks,window_auto_coor,window_minima,window_maxima = get_emg_time_features(windowed_frames)

    #Frequency domain features
    window_mean_pwr, window_pow_peaks, window_tot_pw, window_pow_var,window_fr_minima,window_fr_maxima = get_emg_freq_features(windowed_frames, 512)

    #create feature vector
    feature_matrix = np.array([window_gc,window_zc,window_len,window_rms,window_mean,window_var,window_ssi,window_iemg,window_peaks,window_minima,window_maxima,window_mean_pwr, window_pow_peaks, window_tot_pw, window_pow_var,window_fr_minima,window_fr_maxima])
    feature_matrix = np.transpose(feature_matrix)

    #think of adding mfcc,other transforms as well if needed
    return feature_matrix

def get_emg_time_features(frames):
    window_gc = []
    window_zc = []
    window_len = []
    window_rms = []
    window_mean = []
    window_var = []
    window_ssi = []
    window_iemg = []
    window_peaks = []
    window_auto_coor = []
    window_minima = []
    window_maxima = []

    for frame in frames:
        #gradient
        gs = np.gradient(frame)
        signs = np.sign(gs)
        sign_change = 0
        last_sign = 0

        for sign in signs:
            if last_sign == 1 and sign == -1:
                sign_change += 1
                last_sign = sign
            elif last_sign == -1 and sign == 1:
                sign_change += 1
                last_sign = sign
            elif last_sign == 0:
                last_sign = sign
        window_gc.append(sign_change)

        #zero crossing
        zero_crossings = np.where(np.diff(np.sign(frame)))[0]
        window_zc.append(len(zero_crossings))

        #window length
        sum = 0
        for x in range(0, len(frame) - 2):
            sum += np.absolute(frame[x + 1] - frame[x])
        window_len.append(sum)

        #rms
        rms = math.sqrt(np.sum(np.square(frame)) / len(frame))
        window_rms.append(rms)

        #mean
        m = np.mean(frame)
        window_mean.append(m)

        #variance
        var = np.var(frame)
        window_var.append(var)

        #ssi
        ssi = np.sum(np.square(frame))
        window_ssi.append(ssi)

        #iemg
        sum = np.sum(np.absolute(frame))
        window_iemg.append(sum)

        #peaks
        peakind = signal.find_peaks_cwt(frame, np.arange(1, 10))
        window_peaks.append(len(peakind))

        #auto coorealtion
        freqs = np.fft.rfft(frame)
        auto1 = freqs * np.conj(freqs)
        auto2 = auto1 * np.conj(auto1)
        result = np.fft.irfft(auto2)
        window_auto_coor.append(result)

        # minima
        minima = np.amin(frame)
        window_minima.append(minima)

        # maxima
        maxima = np.amax(frame)
        window_maxima.append(maxima)

    return np.array(window_gc),np.array(window_zc),np.array(window_len),np.array(window_rms),np.array(window_mean),np.array(window_var),np.array(window_ssi),np.array(window_iemg),np.array(window_peaks),np.array(window_auto_coor),np.array(window_minima),np.array(window_maxima)



#find mean frequency
#find power frequency ratio
#peak frequencies
def get_emg_freq_features(frames,NFFT):
    window_mean_pwr = []
    window_n_peaks = []
    window_tot_pw = []
    window_pow_var = []
    window_minima = []
    window_maxima = []

    frames_pw_spec = sigproc.powspec(frames,NFFT)
    for frame in frames_pw_spec:
        #n_peaks
        peakind = signal.find_peaks_cwt(frame, np.arange(1, 10))
        window_n_peaks.append(len(peakind))

        #mean
        m = np.mean(frame)
        window_mean_pwr.append(m)

        #total power
        sum = np.sum(np.absolute(frame))
        window_tot_pw.append(sum)

        #power variance
        var = np.var(frame)
        window_pow_var.append(var)

        # minima
        minima = np.amin(frame)
        window_minima.append(minima)

        # maxima
        maxima = np.amax(frame)
        window_maxima.append(maxima)

    return np.array(window_mean_pwr),np.array(window_n_peaks),np.array(window_tot_pw),np.array(window_pow_var),np.array(window_minima),np.array(window_maxima)

def gr_change(frames):
    window_gc = []
    for frame in frames:
        gs = np.gradient(frame)
        signs = np.sign(gs)
        sign_change = 0
        last_sign = 0
        for sign in signs:
            if last_sign == 1 and sign == -1:
                sign_change += 1
                last_sign = sign
            elif last_sign == -1 and sign == 1:
                sign_change +=1
                last_sign = sign
            elif last_sign == 0:
                last_sign = sign
        #print(sign_change)
        window_gc.append(sign_change)
    return np.array(window_gc)

def zero_crossings(frames):
    window_zc = []
    for frame in frames:
        zero_crossings = np.where(np.diff(np.sign(frame)))[0]
        window_zc.append(len(zero_crossings))
    return np.array(window_zc)

def find_waveform_length(frames):
    window_len = []
    for frame in frames:
        sum = 0
        for x in range(0, len(frame)-2):
            sum += np.absolute(frame[x+1] - frame[x])
        window_len.append(sum)
    return np.array(window_len)

def find_rms(frames):
    window_rms = []
    for frame in frames:
        rms = math.sqrt(np.sum(np.square(frame))/len(frame))
        window_rms.append(rms)
    return np.array(window_rms)



def find_mean(frames):
    window_mean = []
    for frame in frames:
        m = np.mean(frame)
        window_mean.append(m)
    return np.array(window_mean)

def find_var(frames):
    window_var = []
    for frame in frames:
        var = np.var(frame)
        window_var.append(var)
    return np.array(window_var)

def find_ssi(frames):
    window_ssi = []
    for frame in frames:
        ssi = np.sum(np.square(frame))
        window_ssi.append(ssi)
    return np.array(window_ssi)

def iemg(frames):
    window_iemg = []
    for frame in frames:
        sum = np.sum(np.absolute(frame))
        window_iemg.append(sum)
    return np.array(window_iemg)


def find_peaks(frames):
    window_peaks = []
    for frame in frames:
        peakind = signal.find_peaks_cwt(frame, np.arange(1, 10))
        window_peaks.append(len(peakind))
    return np.array(window_peaks)

def get_sliding_windows(data):
    ovlp_windows = sigproc.framesig(data,window_size,overlap_size)
    return ovlp_windows

def windowfn(frames):
    window_norm_frames = []
    for frame in frames:
        window = signal.hamming(len(frame))
        window_norm_frames.append(window*frame)
    #print(np.array(window_norm_frames))
    return np.array(window_norm_frames)


def plot_frames(frames):
    fig, axs = plt.subplots(frames.shape[0], sharex=True, sharey=True)
    for i, (ax, frame) in enumerate(zip(axs, frames)):
        ax.set_title("{0}th frame".format(i))
        ax.plot(frame)
        ax.grid(True)
    #for frame in frames:

def estimated_autocorrelation(frames):
    window_auto_coor = []
    for x in frames:
        n = len(x)
        variance = x.var()
        x = x-x.mean()
        r = np.correlate(x, x, mode = 'full')[-n:]
        #assert N.allclose(r, N.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
        result = r/(variance*(np.arange(n, 0, -1)))
        window_auto_coor.append(result)
    return np.array(window_auto_coor)
