from spiking_radio_reservoir import *
from utils.dataset import load_dataset
from utils.modulator import AsynchronousDeltaModulator, modulate
from brian2 import us

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
    'num_samples': 20,
    'time_sample': np.arange(128),
    'thrup': 0.01,
    'thrdn': 0.01,
    'resampling_factor': 200,
    'stretch_factor': 50,
    'pause': 5000
}
tot_num_samples = settings['num_samples']*len(settings['modulations'])
dataset = load_dataset('./data/radioML/RML2016.10a_dict.pkl', snr=settings['snr'], normalize=True)
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
stimulation = (len(settings['time_sample'])*settings['resampling_factor']*settings['stretch_factor']/1e3)*ms
duration = (stimulation+settings['pause']*ms)*settings['num_samples']*len(settings['modulations'])
to = 0.0*ms
for (i, mod) in tqdm(enumerate(settings['modulations'])):
    for j in range(settings['num_samples']):
        sample = dataset[(mod, settings['snr'])][j]
        ix, tx, _, _ = modulate(modulator[0], modulator[1], settings['time_sample'], sample, \
                                resampling_factor=settings['resampling_factor'], \
                                stretch_factor=settings['stretch_factor'])
        tx = tx*us + to
        indices.extend(ix)
        times.extend(tx)
        Y.append(i)
        to = (stimulation+settings['pause']*ms)*(i*settings['num_samples']+j+1)
print("\t * total duration: {}s".format(duration))

# Create experiment folder
exp_name = 'mod_rec_hennequin'
exp_dir = './experiments/{}-{}'.format(exp_name, datetime.now().strftime("%Y-%m-%dT%H-%M"))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

# Define reservoir parameters
params = {
    'wGen': 3500,
    'wInp': 3500,
    'pIR': 0.1,
    'pE_local': 0.5,
    'pI_local': 1.0,
    'k': 3,
    'DoC': 0.2,
    'loc_wResE': 1000,
    'scale_wResE': 100,
    'loc_wResI': -1000,
    'scale_wResI': 100,
    'N': 200,
    'tau': (20, 200),
    'Ngx': 10,
    'Ngy': 10,
}

# Plots
plot_flags = {
    'raster': False,
    'result': True,
    'network': True,
    'weights': True,
    'weights3D': False,
    'similarity': True,
    'currents': False
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

# Setup connectivity of the network
connectivity = setup_hennequin_connectivity(params['N'], params['pIR'], params['Ngx'], params['Ngy'], \
    params['pE_local'], params['pI_local'], params['k'], params['DoC'], \
    params['loc_wResE'], params['scale_wResE'], params['loc_wResI'], params['scale_wResI'])

# Run experiment
score = experiment(wGen=params['wGen'], wInp=params['wInp'], connectivity=connectivity, \
    N=params['N'], tau=params['tau'], Ngx=params['Ngx'], Ngy=params['Ngy'], \
    indices=indices, times=times, stretch_factor=settings['stretch_factor'], \
    duration=duration, ro_time=stimulation+settings['pause']*ms, \
    modulations=settings['modulations'], snr=settings['snr'], num_samples=settings['num_samples'], Y=Y, \
    plot=plot_flags, store=False, title=exp_name, exp_dir=exp_dir, remove_device=True)
print(score)