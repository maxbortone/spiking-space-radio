import numpy as np
import pickle
from itertools import product
from sklearn.preprocessing import Normalizer


def load_dataset(path, modulations='all', snr='all', scale=None, normalize=False):
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
    if scale:
        for c in classes:
            dataset[c] = scale*dataset[c]
    return dataset, classes