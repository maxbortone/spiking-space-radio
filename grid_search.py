from spiking_radio_reservoir import *
from utils.dataset import load_dataset
from utils.reservoir import getTauCurrent, getAhpTauCurrent
from utils.modulator import AsynchronousDeltaModulator, modulate

np.random.seed(42)

# Set brian2 extra compilation arguments
prefs.devices.cpp_standalone.extra_make_args_unix = ["-j6"]

# Import dataset and prepare samples
# - modulations:
#       '8PSK', 'BPSK', 'QPSK',
#       'QAM16', 'QAM64',      
#       'CPFSK', 'GFSK',      
#       'AM-DSB', 'AM-SSB',
#       'PAM4', 'WBFM'
print("- Importing dataset")
settings = {
    'snr': 18,
    'modulations': ['8PSK', 'BPSK', 'QPSK'],
    'scale': 50,
    'num_samples': 2,
    'time_sample': np.arange(128),
    'thrup': 0.1,
    'thrdn': 0.1,
    'resampling_factor': 200,
    'stretch_factor': 1000,
    'stop_after': 10000,
    'stop_neuron': 4,
    'pause': 500
}
tot_num_samples = settings['num_samples']*len(settings['modulations'])
dataset, _ = load_dataset('./data/radioML/RML2016.10a_dict.pkl', snr=settings['snr'], scale=settings['scale'])
# Define delta modulators
modulator = [
    AsynchronousDeltaModulator(settings['thrup'], settings['thrdn'], settings['resampling_factor']),
    AsynchronousDeltaModulator(settings['thrup'], settings['thrdn'], settings['resampling_factor'])
]
# Prepare stimulus
print("- Preparing input stimulus")
indices = []
times = []
Y = []
stimulation = (len(settings['time_sample'])*settings['stretch_factor']/1e3)*ms
duration = (stimulation+settings['pause']*ms)*settings['num_samples']*len(settings['modulations'])
to = 0.0*ms
for (i, mod) in tqdm(enumerate(settings['modulations'])):
    for j in range(settings['num_samples']):
        sample = dataset[(mod, settings['snr'])][j]
        ix, tx, _, _ = modulate(modulator[0], modulator[1], settings['time_sample'], sample, \
                                resampling_factor=settings['resampling_factor'], \
                                stretch_factor=settings['stretch_factor'])
        tx = tx*us + to
        stop_time = tx[-1]+settings['stop_after']*ms
        indices.extend(ix)
        indices.append(settings['stop_neuron'])
        times.extend(tx)
        times.append(stop_time)
        Y.append(i)
        to = (stimulation+settings['pause']*ms)*(i*settings['num_samples']+j+1)
Y = np.array(Y)
print("\t * total duration: {}s".format(duration))

# Create experiment folder
exp_name = 'grid_search'
exp_dir = './experiments/{}-{}'.format(exp_name, datetime.now().strftime("%Y-%m-%dT%H-%M"))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

# Define reservoir parameters
params = {
    'wGen': 500,
    'wInp': 500,
    'pIR': 0.07,
    'pE_local': 0.5,
    'pI_local': 1.0,
    'k': 3,
    'DoC': 0.2,
    'loc_wResE': 120,
    'scale_wResE': 20,
    'loc_wResI': -200,
    'scale_wResI': 40,
    'Ninp': 4,
    #'N': 800,
    #'Ngx': 20,
    #'Ngy': 20
}
# Set currents
params['currents'] = {
    'gInp': {
        'Iahp': 0.5*pA,
        'Itau': getTauCurrent(5*ms),
        'Ispkthr': 0.2*nA
    },
    'gRes': {
        'Iahp': 0.5*pA,
        # 'Itauahp': getAhpTauCurrent(50*ms),
        'Itau': getTauCurrent(20*ms),
        'Ispkthr': 0.2*nA
    },
    'sInpRes': {
        'Ie_tau': getTauCurrent(7*ms)
    },
    'sResRes': {
        'Ie_tau': getTauCurrent(7*ms),
        'Ii_tau': getTauCurrent(7*ms)
    }
}
# Set mismatch
params['mismatch'] = {
    'gRes': {
        'Itau': 0.1
    },
    'sResRes': {
        'Ie_tau': 0.1,
        'Ii_tau': 0.1,
    }
}

# Plots
plot_flags = {
    'raster': False,
    'result': False,
    'network': False,
    'weights': False,
    'weights3D': False,
    'similarity': False,
    'currents': False,
    'accuracy': False
}

# Set parameter grid
grid_params = {
    'N': [200],#[200, 300, 400, 500, 600, 700, 800, 900, 1000]
    'Ngx': [10],
    'Ngy': [10]
}
grid = zip(grid_params['N'], grid_params['Ngx'], grid_params['Ngy'])

# Store all the parameters and settings
settings_path = exp_dir + '/conf.txt'
with open(settings_path, 'w+') as f:
    f.write('Model parameters: \n')
    for (key, value) in params.items():
        f.write('- {}: {}\n'.format(key, value))
    f.write('Preprocessing settings: \n')
    for (key, value) in settings.items():
        f.write('- {}: {}\n'.format(key, value))
    f.write('Grid parameters: \n')
    for (key, value) in grid_params.items():
        f.write('- {}: {}\n'.format(key, value))

# Run grid search
for (n, ngx, ngy) in grid:
    print("- Running with: N={}, Ngx={}, Ngy={}".format(n, ngx, ngy))
    title = exp_name + "_{}".format(n)
    # Setup connectivity of the network
    connectivity = setup_hennequin_connectivity(n, params['pIR'], ngx, ngy, \
        params['pE_local'], params['pI_local'], params['k'], params['DoC'], \
        params['loc_wResE'], params['scale_wResE'], params['loc_wResI'], params['scale_wResI'])
    # Run experiment
    score = experiment(wGen=params['wGen'], wInp=params['wInp'], connectivity=connectivity, mismatch=params['mismatch'], \
        N=n, Ninp=params['Ninp'], currents=params['currents'], Ngx=ngx, Ngy=ngy, \
        indices=indices, times=times, stretch_factor=settings['stretch_factor'], \
        duration=duration, ro_time=stimulation+settings['pause']*ms, \
        modulations=settings['modulations'], snr=settings['snr'], num_samples=settings['num_samples'], Y=Y, \
        plot=plot_flags, store=True, title=title, exp_dir=exp_dir, dt=50*us, remove_device=True)
    print(score)