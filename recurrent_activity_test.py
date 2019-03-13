from spiking_radio_reservoir import *
from utils.modulator import AsynchronousDeltaModulator
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

def test(wGen=3500, wInp=3500, loc_wResE=50, scale_wResE=10, loc_wResI=50, scale_wResI=10, \
        pIR=0.3, pInh=0.2, AoC=[0.3, 0.2, 0.5, 0.1], DoC=2, \
        N=200, tau=20, Ngx=5, Ngy=5, Ngz=8, \
        stretch_factor=None, duration=None, ro_time=None, phases=None, num_samples=None, Y=None):
    start = time.perf_counter()
    print("- running with: wInp={}, loc_wResE={}, scale_wResE={}, loc_wResI={}, scale_wResI={}"
        .format(wInp, loc_wResE, scale_wResE, loc_wResI, scale_wResI))
    connectivity = setup_schliebs_connectivity(N, pInh, pIR, Ngx, Ngy, Ngz, AoC, DoC, loc_wResE, scale_wResE, loc_wResI, scale_wResI)
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
    plot_result(X, Y, bins, edges, phases, 18, directory=plots_dir)
    plot_network(network, N, connectivity['res_res']['w'], directory=plots_dir)
    plot_similarity(X, Y, phases, directory=plots_dir)
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

test(loc_wResE=400, scale_wResE=100, loc_wResI=-1000, scale_wResI=100, pIR=0.1, pInh=0.2, AoC=[0.3, 0.2, 0.5, 0.1], DoC=2, 
    stretch_factor=stretch_factor, duration=to, ro_time=duration, phases=phases, num_samples=len(phases), Y=Y)