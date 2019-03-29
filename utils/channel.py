import numpy as np
from itertools import product
from scipy.signal import firwin, lfilter
from commpy.modulation import PSKModem, QAMModem
from commpy.filters import rrcosfilter
from utils.modulator import AsynchronousDeltaModulator

def _round_complex(x):
    return np.round(x.real, decimals=2)+np.round(x.imag, decimals=2)*1j

def send(input_string, m=2, modem=PSKModem, Ts=1e-3, Fs=int(6e4), fc=int(3e3)):
    # convert string to bitsarray
    # pylint: disable=too-many-format-args
    bits_string = ''.join('{0:08b}'.format(x, 'b') for x in bytearray(input_string, encoding='utf-8'))
    input_bits = np.array([int(b) for b in bits_string])
    # instantiate modem
    modem = modem(m)
    # FIX: rounding error in CommPy
    modem.constellation = _round_complex(np.array(modem.constellation))
    # modulate input bits
    dk = modem.modulate(input_bits)
    N = len(dk)
    # weight baseband symbols with Dirac comb function
    ups = int(Ts*Fs)
    x = np.zeros(ups*N, dtype='complex')
    x[::ups] = dk
    # set pulse shaping filter
    t0 = 3*Ts
    _, rrc = rrcosfilter(N=int(2*t0*Fs), alpha=1,Ts=Ts, Fs=Fs)
    # convolve pulse shaping filter with Dirac comb
    u = np.convolve(x, rrc)
    time_sample = np.arange(len(u))/Fs
    # define up-converted I and Q components
    i = u.real
    q = u.imag
    iup = i * np.cos(2*np.pi*time_sample*fc)  
    qup = q * -np.sin(2*np.pi*time_sample*fc)
    # define transmitted signal
    s = iup + qup
    # TODO: apply channel effects
    # .....
    # define down-converted I and Q components
    idown = s * np.cos(2*np.pi*-fc*time_sample) 
    qdown = s * -np.sin(2*np.pi*fc*time_sample)
    # set lowpass filter for image rejection
    BN = 1/(2*Ts)
    cutoff = 5*BN
    lowpass_order = 51   
    lowpass_delay = (lowpass_order//2)/Fs
    lowpass = firwin(lowpass_order, cutoff/(Fs/2))
    # lowpass filter down-converted I and Q components
    idown_lp = lfilter(lowpass, 1, idown)
    qdown_lp = lfilter(lowpass, 1, qdown)
    # define sample
    sample = np.empty((2, len(time_sample)))
    sample[0, :] = idown_lp
    sample[1, :] = qdown_lp
    # define labels
    label_ids = list(map(lambda s: np.where(modem.constellation==s)[0][0], dk))
    labels = np.zeros(ups*N)
    bins = np.zeros(N+1)
    for (i, y) in enumerate(label_ids):
        labels[i*ups:(i+1)*ups] = y
        bins[i+1] = (i+1)*Ts
    delay = t0+lowpass_delay
    time_labels = np.arange(len(labels))/Fs+delay-Ts/2
    bins += delay-Ts/2
    return time_sample, sample, time_labels, labels, bins