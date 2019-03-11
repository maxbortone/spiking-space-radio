import hashlib
from spiking_radio_reservoir import *
from utils.modulator import AsynchronousDeltaModulator
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

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
        tx = tx + to
        indices.extend(ix)
        times.extend(tx)
        Y.append(i)
        to = (stimulation+settings['pause']*ms)*(i*settings['num_samples']+j+1)
print("\t * total duration: {}s".format(duration))

# Create experiment folder
exp_name = 'weights_bo'
exp_dir = './experiments/{}-{}'.format(exp_name, datetime.now().strftime("%Y-%m-%dT%H-%M"))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

# Plots
plot_flags = {
    'raster': False,
    'result': True,
    'network': False,
    'weights': False,
    'similarity': True,
    'currents': False
}

# Define Bayesian optmization process
params = {
    'pIR': 0.1,
    'pInh': 0.2,
    'AoC': [0.3, 0.2, 0.5, 0.1],
    'DoC': 2,
    'N': 200,
    'tau': 20,
    'Ngx': 5,
    'Ngy': 5,
    'Ngz': 8,
}
pbounds = {
    #'N': (50, 200),            # number of neurons   
    #'tau': (20, 200),          # DPI time constant [ms]
    #'pRR': (0.3, 1.0)          # probability of connection
    #'wGen': (50, 5000),         # units of baseweight
    #'wInp': (50, 5000),         # units of baseweight
    'loc_wResE': (1000, 1200),         # units of baseweight
    'scale_wResE': (50, 100),         # units of baseweight
    'loc_wResI': (1000, 1200),         # units of baseweight
    'scale_wResI': (50, 100),         # units of baseweight    
    #'DoC': ()                  # density of connection   
}

def bo_process(loc_wResE=1000, scale_wResE=50, loc_wResI=1000, scale_wResI=50):
    s = 'loc_wResE{}scale_wResE{}'.format(loc_wResE, scale_wResE)
    uid = hashlib.md5(s.encode('utf-8')).hexdigest()
    title = '{}_{}'.format(exp_name, uid)
    return experiment(wGen=3500, wInp=3500, loc_wResE=loc_wResE, scale_wResE=scale_wResE, loc_wResI=loc_wResI, scale_wResI=scale_wResI,
        pIR=params['pIR'], pInh=params['pInh'], AoC=params['AoC'], DoC=params['DoC'], \
        N=params['N'], tau=params['tau'], Ngx=params['Ngx'], Ngy=params['Ngy'], Ngz=params['Ngz'], \
        indices=indices, times=times, stretch_factor=settings['stretch_factor'], \
        duration=duration, ro_time=stimulation+settings['pause']*ms, \
        modulations=settings['modulations'], snr=settings['snr'], num_samples=settings['num_samples'], Y=Y, \
        plot=plot_flags, store=False, title=title, exp_dir=exp_dir, remove_device=True)

optimizer = BayesianOptimization(
    f=bo_process,
    pbounds=pbounds,
    random_state=42
)

# Store all the parameters and settings
settings_path = exp_dir + '/conf.txt'
with open(settings_path, 'w+') as f:
    f.write('Model parameters: \n')
    for (key, value) in params.items():
        f.write('- {}: {}\n'.format(key, value))
    f.write('Optimized parameters: \n')
    for (key, value) in pbounds.items():
        f.write('- {}: {}\n'.format(key, value))
    f.write('Preprocessing settings: \n')
    for (key, value) in settings.items():
        f.write('- {}: {}\n'.format(key, value))

# Subscribe logger
log_path = exp_dir + '/logger.json'
logger = JSONLogger(path=log_path)
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

# TODO: try multithreading
# Optimize model performance
print("- Starting optimization")
start = time.perf_counter()
optimizer.maximize(
    init_points=2,
    n_iter=10,
    kappa=3
)

# Print results
best = optimizer.max
print("- Finished! Best solution: ")
print("\t - score: {}".format(best['target']))
for (key, value) in best['params'].items():
    print("\t - {}: {}".format(key, value))
s = 'loc_wResE{}scale_wResE{}'.format(best['params']['loc_wResE'], best['params']['scale_wResE'])
uid = hashlib.md5(s.encode('utf-8')).hexdigest()
print("\t - uid: {}".format(uid))


# Print runtime info
print("[optimization took: {}]".format(time.perf_counter()-start))