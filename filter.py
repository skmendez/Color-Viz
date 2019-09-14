from scipy.signal import butter, lfilter


def butter_bandpass(highcut, fs, order=5):
    b, a = butter(order, highcut, btype='lowpass', fs=fs)
    return b, a


def butter_bandpass_filter(data, highcut, fs, order=5):
    b, a = butter_bandpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
