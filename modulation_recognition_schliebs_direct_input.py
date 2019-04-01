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
    'num_samples': 20,
    'time_sample': np.arange(128),
    'thrup': 0.01,
    'thrdn': 0.01,
    'resampling_factor': 200,
    'stretch_factor': 50,
    'stop_after': 10000,
    'stop_neuron': 4,
    'pause': 500
}
tot_num_samples = settings['num_samples']*len(settings['modulations'])
dataset = load_dataset('./data/radioML/RML2016.10a_dict.pkl', snr=settings['snr'], scale=settings['scale'])
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
exp_name = 'mod_rec_schliebs_direct_input'
exp_dir = './experiments/{}-{}'.format(exp_name, datetime.now().strftime("%Y-%m-%dT%H-%M"))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

# Define reservoir parameters
params = {
    'wInp': 500,
    'pIR': 0.1,
    'pInh': 0.2,
    'AoC': [0.3, 0.2, 0.5, 0.1],
    'DoC': 2,
    'loc_wResE': 1,
    'scale_wResE': 0.5,
    'loc_wResI': -1,
    'scale_wResI': 0.5,
    'Ninp': 4,
    'N': 800,
    'Ngx': 10,
    'Ngy': 10,
    'Ngz': 8,
}

# Plots
plot_flags = {
    'raster': False,
    'result': True,
    'network': True,
    'weights': True,
    'weights3D': True,
    'similarity': True,
    'currents': True,
    'accuracy': True
}

# Setup connectivity of the network
connectivity = setup_schliebs_connectivity(params['N'], params['pInh'], params['pIR'], \
    params['Ngx'], params['Ngy'], params['Ngz'], params['AoC'], params['DoC'], \
    params['loc_wResE'], params['scale_wResE'], params['loc_wResI'], params['scale_wResI'])

# Set currents
num_syn = len(connectivity['res_res']['w'])
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

# Store all the parameters and settings
settings_path = exp_dir + '/conf.txt'
with open(settings_path, 'w+') as f:
    f.write('Model parameters: \n')
    for (key, value) in params.items():
        f.write('- {}: {}\n'.format(key, value))
    f.write('Preprocessing settings: \n')
    for (key, value) in settings.items():
        f.write('- {}: {}\n'.format(key, value))

# Run experiment
score = experiment(wGen=None, wInp=params['wInp'], connectivity=connectivity, mismatch=params['mismatch'], \
    N=params['N'], Ninp=params['Ninp'], currents=params['currents'], Ngx=params['Ngx'], Ngy=params['Ngy'], Ngz=params['Ngz'], \
    direct_input=True, indices=indices, times=times, stretch_factor=settings['stretch_factor'], \
    duration=duration, ro_time=stimulation+settings['pause']*ms, \
    modulations=settings['modulations'], snr=settings['snr'], num_samples=settings['num_samples'], Y=Y, \
    plot=plot_flags, store=False, title=exp_name, exp_dir=exp_dir, dt=50*us, remove_device=True)
print(score)