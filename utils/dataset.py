import numpy as np
import pickle
from itertools import product
from sklearn.preprocessing import Normalizer
from brian2 import us


def load_dataset(path, modulations='all', snr='all', normalize=False):
    with open(path, 'rb') as f:
        dataset = pickle.load(f, encoding='latin1')
    all_modulations = np.unique(list(map(lambda x: x[0], dataset.keys())))
    all_snr = np.unique(list(map(lambda x: x[1], dataset.keys())))
    if type(snr) is int:
        snr = [snr]
    if type(modulations) is str and not modulations=='all':
        modulations = [modulations]
    if not modulations=='all' and not snr=='all':
        classes = list(product(modulations, snr))
    elif not modulations=='all' and snr=='all':
        classes = list(product(modulations, all_snr))
    elif modulations=='all' and not snr=='all':
        classes = list(product(all_modulations, snr))
    else:
        classes = list(product(all_modulations, all_snr))
    if not modulations=='all' or not snr=='all':
        dataset = dict((c, dataset[c]) for c in classes)
    if normalize:
        for c in classes:
            samples = dataset[c]
            samples[:, 0, :] = Normalizer(copy=False).fit_transform(samples[:, 0, :])
            samples[:, 1, :] = Normalizer(copy=False).fit_transform(samples[:, 1, :])
    return dataset
    
def modulate(admI, admQ, time, sample, resampling_factor=1, stretch_factor=1):
    admI.interpolate(time, sample[0, :])
    admQ.interpolate(time, sample[1, :])
    admI.encode()
    admQ.encode()
    indices = []
    times = []
    time_stim = np.arange(len(time)*resampling_factor)*stretch_factor
    for i in range(admI.time_length):
        if admI.up[i]:
            indices.append(0)
            times.append(time_stim[i])
        if admI.dn[i]:
            indices.append(1)
            times.append(time_stim[i])
        if admQ.up[i]:
            indices.append(2)
            times.append(time_stim[i])
        if admQ.dn[i]:
            indices.append(3)
            times.append(time_stim[i])
    signal = np.array([admI.vin, admQ.vin])
    return np.array(indices), np.array(times)*us, time_stim, signal