from itertools import chain

import numpy as np
from python_speech_features import mfcc
from python_speech_features import ssc
from FeatureExtraction_Kumar import TimeDomain


# https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py
# https://softwareengineering.stackexchange.com/questions/182093/why-store-a-function-inside-a-python-dictionary
# https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#mean_abs_change
# feat_names will consists of names of the features that has to be computed
############ Frequency domain ##########
# Useful for signals with periodicities
# While time-domain analysis shows how a signal changes over time, frequency-domain analysis shows how the signal's energy is distributed
# over a range of frequencies. A frequency-domain representation also includes information on the phase shift that must be applied to each frequency component in order to recover the original time signal with a combination of all the individual frequency components.
# A signal can be converted between the time and frequency domains with a pair of mathematical operators called a transform.
# An example is the Fourier transform, which decomposes a function into the sum of a (potentially infinite) number of sine wave
# frequency components. The 'spectrum' of frequency components is the frequency domain representation of the signal.
# The inverse Fourier transform converts the frequency domain function back to a time function.
# The fft and ifft functions in MATLAB allow you to compute the Discrete Fourier transform (DFT) of a signal and
# the inverse of this transform respectively.
# Spectral Entropy - Measure of the distribution of frequency components

def fft_coeff(X, num_bins):
    coefficients = np.fft.fft(X)
    mag_coefficients = abs(coefficients)
    # When the input a is a time-domain signal and A = fft(a), np.abs(A) is its amplitude spectrum and np.abs(A)**2 is its power spectrum. The phase spectrum is obtained by np.angle(A).
    feature_names = []
    ff_coeff_feat = []
    # ff_coeff_feat.append(TimeDomain.amean(mag_coefficients))
    # feature_names.append('fftc_amean') # Arithmatic mean
    # ff_coeff_feat.append(TimeDomain.abs_energy(mag_coefficients))
    # feature_names.append('fftc_abs_energy') # absolute energy
    # ff_coeff_feat.append(TimeDomain.cid_ce(mag_coefficients)) ## This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
    # feature_names.append('fftc_cid_ce')
    ff_coeff_feat.append(TimeDomain.fquantile(mag_coefficients))
    feature_names.append('fftc_fquantile')
    # ff_coeff_feat.append(TimeDomain.number_peaks(mag_coefficients))
    # feature_names.append('fftc_number_peaks')
    ff_coeff_feat.append(TimeDomain.squantile(mag_coefficients))
    feature_names.append('fftc_squantile')
    ff_coeff_feat.append(TimeDomain.std_dev(mag_coefficients))
    feature_names.append('fftc_std_dev')
    ff_coeff_feat.append(TimeDomain.tquantile(mag_coefficients))
    feature_names.append('fftc_tquantile')
    return feature_names,ff_coeff_feat


# Also known as differential and acceleration coefficients. The MFCC feature vector describes only the power spectral
# envelope of a single frame, but it seems like speech would also have information in the dynamics i.e. what are the
# trajectories of the MFCC coefficients over time. It turns out that calculating the MFCC trajectories and appending
# them to the original feature vector increases ASR performance by quite a bit (if we have 12 MFCC coefficients,
# we would also get 12 delta coefficients, which would combine to give a feature vector of length 24).
def mfcc_features(signal, samplerate, winlen, winstep, fftlength):
    """Compute MFCC features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the samplerate of the signal we are working with.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        :param numcep: the number of cepstrum to return, default 13
        :param nfilt: the number of filters in the filterbank, default 26.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
        :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
        :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
        """
    features = mfcc(signal, samplerate=samplerate, winlen=winlen, winstep=winstep, numcep=13, nfilt=26, nfft=fftlength, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True)
    fnames = []
    numcep = 13 #
    for i in range(numcep):
        fnames.append('mfcc_'+str(i))
    if features.shape[1] != numcep:
     raise ValueError(f'Number of features should be equal to {numcep}')
    return fnames, features[0].tolist()


def ssc_features(signal, samplerate, winlen, winstep, fftlength):
    """Compute Spectral Subband Centroid features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the samplerate of the signal we are working with.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        :param nfilt: the number of filters in the filterbank, default 26.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
        :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
        """
    features = ssc(signal, samplerate=samplerate, winlen=winlen, winstep=winstep, nfilt=26, nfft=fftlength, lowfreq=0, highfreq=None, preemph=0.97,
                   winfunc=lambda x: np.ones((x,)))
    nfilt = 26 # default
    fnames = []
    for i in range(nfilt):
        fnames.append('ssc_'+str(i))
    if features.shape[1] != nfilt:
     raise ValueError(f'Number of features should be equal to nfilt:{nfilt}')

    return fnames, features[0].tolist()

def getall_fd_features(frame, num_bins, samplerate, winlen, winstep, fftlength):
    fftnames, fftfv = fft_coeff(frame, num_bins) # Just extracting this for the moment
    # mfccnames, mfccfv = mfcc_features(frame, samplerate, winlen, winstep, fftlength)
    # sscnames, sscfv = ssc_features(frame, samplerate, winlen, winstep, fftlength)
    # freqfvs = fftfv + mfccfv + sscfv
    # feature_names = fftnames+mfccnames+sscnames
    # return feature_names, freqfvs

    return fftnames, fftfv
