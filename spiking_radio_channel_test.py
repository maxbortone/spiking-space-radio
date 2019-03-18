import matplotlib.pyplot as plt
from utils.channel import send
from utils.modulator import AsynchronousDeltaModulator, modulate
from utils.plotting import plot_sample_with_labels

input_string = "neuro"
time_sample, sample, time_labels, labels, bins = send(input_string, m=8)
settings = {
    'thrup': 0.1,
    'thrdn': 0.1,
    'resampling_factor': 10
}
modulators = [
    AsynchronousDeltaModulator(settings['thrup'], settings['thrdn'], settings['resampling_factor']),
    AsynchronousDeltaModulator(settings['thrup'], settings['thrdn'], settings['resampling_factor'])
]
indices, times, time_stim, signal = modulate(modulators[0], modulators[1], time_sample, sample, resampling_factor=settings['resampling_factor'])
plot_sample_with_labels(signal, time_stim, indices, times, labels, time_labels, bins, figsize=(12, 9))
plt.show()