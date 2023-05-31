import math

import numpy as np
from scipy.signal import butter, filtfilt


def up_down_sampling_and_fs_filtering(timedXYZ, sourceFs, requiredFs):
    if sourceFs != requiredFs:
        if sourceFs > 2 * requiredFs:
            filterFreq = requiredFs / 2
            print('FILTER: Low-pass filter, cut-off @%f Hz...' % filterFreq)
            timedXYZ[:, 1:4] = filter(timedXYZ[:, 1:4], sourceFs, high_freq=filterFreq)
            print(timedXYZ)

        sourceCount = timedXYZ.shape[0]
        destNum = math.floor((requiredFs * sourceCount) // sourceFs)
        indexes = np.linspace(start=0, stop=sourceCount, num=destNum, endpoint=False, dtype=int)
        print('RESAMPLE: Nearest from %f samples @%fHz (%f s) to %f samples @%fHz (%f s)' % (
        sourceCount, sourceFs, sourceCount / sourceFs, destNum, requiredFs, destNum / requiredFs))
        timedXYZ = np.take(timedXYZ, indexes, axis=0)
    else:
        raise "sourceFs can't be the same of requiredFs"
    return timedXYZ


def build_activity_counts_without_matlab(data, epoch):
    """
    te Lindert BH, et al.  Sleep estimates using microelectromechanical systems (MEMS). Sleep. 2013;36(5):781–789.
    :param data: acceleration data, it is a numpy array with 4 columns: time, x, y, z
    :param epoch: epoch in seconds, e.g. epoch=30 means that the function will return the activity counts for 30 seconds
    :return:
    """
    fs = 50
    # time = np.arange(np.amin(data[:, 0]), np.amax(data[:, 0]), 1.0 / fs)
    # z_data = np.interp(time, data[:, 0], data[:, 3])
    z_data = data[:, 3]
    cf_low = 3
    cf_hi = 11
    order = 5
    w1 = cf_low / (fs / 2)
    w2 = cf_hi / (fs / 2)
    pass_band = [w1, w2]
    b, a = butter(order, pass_band, 'bandpass')

    z_filt = filtfilt(b, a, z_data)
    z_filt = np.abs(z_filt)
    top_edge = 5
    bottom_edge = 0
    number_of_bins = 128

    bin_edges = np.linspace(bottom_edge, top_edge, number_of_bins + 1)
    binned = np.digitize(z_filt, bin_edges)

    counts = max2epochs(binned, fs, epoch)
    counts = (counts - 18) * 3.07  # why only minus 18 ?
    counts[counts < 0] = 0

    time_counts = np.linspace(np.min(data[:, 0]), max(data[:, 0]), np.shape(counts)[0])
    time_counts = np.expand_dims(time_counts, axis=1)
    counts = np.expand_dims(counts, axis=1)
    output = np.hstack((time_counts, counts))
    return output


def max2epochs(data, fs, epoch):
    data = data.flatten()

    seconds = int(np.floor(np.shape(data)[0] / fs))
    data = np.abs(data)
    data = data[0:int(seconds * fs)]

    data = data.reshape(fs, seconds, order='F').copy()

    data = data.max(0)
    data = data.flatten()
    N = np.shape(data)[0]
    num_epochs = int(np.floor(N / epoch))
    data = data[0:(num_epochs * epoch)]

    data = data.reshape(epoch, num_epochs, order='F').copy()
    epoch_data = np.sum(data, axis=0)
    epoch_data = epoch_data.flatten()

    return epoch_data