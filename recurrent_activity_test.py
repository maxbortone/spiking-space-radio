from spiking_radio_reservoir import *
from utils.plotting import plot_sample
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def I(f, t):
    return np.sin(2*np.pi*f*t)

def Q(f, t, p):
    return np.sin(2*np.pi*f*t+p)

def generate_sample(freq, phase, plot=False, title=None):
    sample = np.array([I(freq, time_sample), Q(freq, time_sample, phase)])
    indices, times, time_stim, signal = modulate(modulator[0], modulator[1], time_sample, sample, \
        resampling_factor=resampling_factor, stretch_factor=stretch_factor)
    if plot:
        plot_sample(signal, time_stim, indices, times, title)
    return indices, times

def test(wGen=3500, wInp=3500, loc_wRes=50, scale_wRes=10, pIR=0.3, pInh=0.2, AoC=[0.3, 0.5, 0.1], DoC=2, \
        N=200, tau=20, Ngx=10, Ngy=20, \
        stretch_factor=None, duration=None, ro_time=None, num_samples=None, Y=None):
    start = time.perf_counter()
    print("- running with: wInp={}, loc_wRes={}, scale_wRes={}".format(wInp, loc_wRes, scale_wRes))
    connectivity = setup_connectivity(N, pInh, pIR, Ngx, Ngy, AoC, DoC, loc_wRes, scale_wRes)
    Itau = getTauCurrent(tau*ms)
    # Set C++ backend and time step
    title = 'recact_{}'.format(os.getpid())
    directory = '../brian2_devices/' + title
    set_device('cpp_standalone', directory=directory, build_on_run=True)
    device.reinit()
    device.activate(directory=directory, build_on_run=True)
    defaultclock.dt = stretch_factor*us
    # Initialize network
    network = init_network(indices, times, connectivity, N, Itau, wGen, wInp)    
    # Run simulation
    network.run(duration, recompile=True)
    # Readout activity
    X, bins, edges = readout(network['mRes'], ro_time, N, num_samples, bin_size=1)
    # Plot
    plots_dir = directory+'/plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    for i in range(num_samples):
        fig= plt.figure()
        plt.imshow(X[i].T, interpolation='nearest', origin='low', aspect='auto', \
           extent=[0, bins[0], 0, bins[1]], cmap='viridis')
        plt.title(r"$\phi = {}\pi/4$".format(i))
        plt.xlabel('vector element')
        plt.ylabel('neuron index')
        plt.savefig(plots_dir+'/phi_{}pi4.pdf'.format(i))
        plt.close(fig=fig)
    plot_network(network, N, connectivity['res_res']['w'], directory=plots_dir)
    # Measure reservoir perfomance
    X = list(map(lambda x: x.T.flatten(), X))
    S = cosine_similarity(X)
    fig = plt.figure()
    plt.imshow(S.T, interpolation='nearest', origin='low', aspect='auto', \
           extent=[0, num_samples, 0, num_samples], cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.savefig(plots_dir+'/similarity.pdf')
    plt.close(fig=fig)
    print("- experiment took {} [s]".format(time.perf_counter()-start))
    return

# Set brian2 extra compilation arguments
prefs.devices.cpp_standalone.extra_make_args_unix = ["-j6"]

# Sample generation settings
time_sample = np.arange(128)
resampling_factor = 200
stretch_factor = 50
pause = 2000*ms
stimulation = (len(time_sample)*resampling_factor*stretch_factor/1e3)*ms
duration = stimulation+pause
num_cycles = 4
freq = num_cycles/len(time_sample)
phase = np.pi/4
thrup = 0.1
thrdn = 0.1
modulator = [
    AsynchronousDeltaModulator(thrup, thrdn, resampling_factor),
    AsynchronousDeltaModulator(thrup, thrdn, resampling_factor)
]

# Concatenate small dataset
phases = [i*np.pi/4 for i in range(8)]
indices = []
times = []
to = 0.0*ms
Y = []
for (i, p) in enumerate(phases):
    ix, tx = generate_sample(freq, p)
    tx = tx + to
    indices.extend(ix)
    times.extend(tx)
    to = duration*(i+1)
    Y.append(i)

test(loc_wRes=50, scale_wRes=10, pIR=0.3, pInh=0.2, AoC=[1.0, 1.0, 1.0], DoC=2, 
    stretch_factor=stretch_factor, duration=to, ro_time=duration, num_samples=len(phases), Y=Y)