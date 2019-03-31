import os, joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grs
from datetime import datetime
from brian2 import us, ms, second, nA, pA, prefs, device, set_device, defaultclock
from utils.plotting import plot_stimulus, plot_input_layer_with_membrane_potential
from utils.plotting import plot_hennequin_reservoir_raster, plot_currents_distributions
from utils.plotting import plot_hennequin_reservoir_raster_without_input
from utils.plotting import plot_sample_with_labels
from utils.modulator import AsynchronousDeltaModulator, modulate
from utils.channel import send
from commpy.modulation import PSKModem, QAMModem
from utils.reservoir import getTauCurrent
from spiking_radio_reservoir import setup_hennequin_connectivity
from spiking_radio_reservoir import setup_generator, setup_input_layer, setup_reservoir_layer, init_network

# Set brian2 extra compilation arguments
prefs.devices.cpp_standalone.extra_make_args_unix = ["-j6"]

# Create experiment folder
exp_name = 'readout'
exp_dir = './experiments/{}-{}'.format(exp_name, datetime.now().strftime("%Y-%m-%dT%H-%M"))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

# Stimulus settings
input_strings = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
modulations = [2, 4, 8]
num_samples = len(input_strings)*len(modulations)
settings = {
    'thrup': 0.05,
    'thrdn': 0.05,
    'resampling_factor': 100,
    'stretch_factor': 10,
    'ro_time': 500*ms,
    'ro_bin_size': 10*ms
}
# Define delta modulators
modulators = [
    AsynchronousDeltaModulator(settings['thrup'], settings['thrdn'], settings['resampling_factor']),
    AsynchronousDeltaModulator(settings['thrup'], settings['thrdn'], settings['resampling_factor'])
]
# Prepare stimulus
print("- Preparing input stimulus")
indices = []
times = []
Y = []
to = 0.0*second
i = 1
for input_string in input_strings:
    for modulation in modulations:
        time_sample, sample, time_labels, labels, bins = send(input_string, m=modulation)
        ix, tx, _, _ = modulate(modulators[0], modulators[1], time_sample, sample, \
                                resampling_factor=settings['resampling_factor'], \
                                stretch_factor=settings['stretch_factor'])
        tx = tx*second + to
        indices.extend(ix)
        times.extend(tx)
        Y.append([i, input_string, modulation])
        to = settings['ro_time']*i
        i += 1
indices = np.array(indices)
times = np.array(times)*second
duration = to
print("\t * total duration: {}s".format(duration))
plot_stimulus(indices, times, settings['ro_time'], num_samples)
plt.savefig(exp_dir+"/stimulus.pdf", bbox_inches='tight')

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
    'N': 800,
    'Ngx': 20,
    'Ngy': 20
}

# Setup connectivity of the network
print("- Setting up connectivity")
connectivity = setup_hennequin_connectivity(params['N'], params['pIR'], params['Ngx'], params['Ngy'], \
    params['pE_local'], params['pI_local'], params['k'], params['DoC'], \
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
        'Itau': np.random.normal(loc=getTauCurrent(20*ms)/pA, scale=1, size=params['N'])*pA,
        'Ispkthr': 0.2*nA
    },
    'sInpRes': {
        'Ie_tau': getTauCurrent(7*ms)
    },
    'sResRes': {
        'Ie_tau': np.random.normal(loc=getTauCurrent(7*ms)/pA, scale=1, size=num_syn)*pA,
        'Ii_tau': np.random.normal(loc=getTauCurrent(7*ms)/pA, scale=1, size=num_syn)*pA,
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

# Build network
directory = '../brian2_devices/' + exp_name
set_device('cpp_standalone', directory=directory, build_on_run=True)
device.reinit()
device.activate(directory=directory, build_on_run=True)
defaultclock.dt = 50*us
components = {'generator': None, 'layers': {}, 'synapsis': {}, 'monitors': {}}
components = setup_generator(components)
components = setup_input_layer(components, connectivity, params['Ninp'], params['currents'], params['wGen'])
components = setup_reservoir_layer(components, connectivity, params['N'], params['currents'], params['wInp'])
components['layers']['run_reg_gRes'] = components['layers']['gRes'].run_regularly("Imem=0*pA", dt=settings['ro_time'])
components['synapsis']['run_reg_sResRes'] = components['synapsis']['sResRes'].run_regularly("""
                                                                                                Ie_syn=Io_syn
                                                                                                Ii_syn=Io_syn
                                                                                            """, dt=settings['ro_time'])
network = init_network(components, indices, times)    
network.run(duration, recompile=True)

# Get readout
print("- Getting readout")
input_neurons_Iup = connectivity['inp_res']['j'][connectivity['inp_res']['i']==0]
input_neurons_Idn = connectivity['inp_res']['j'][connectivity['inp_res']['i']==1]
input_neurons_Qup = connectivity['inp_res']['j'][connectivity['inp_res']['i']==2]
input_neurons_Qdn = connectivity['inp_res']['j'][connectivity['inp_res']['i']==3]
input_neurons = np.concatenate([input_neurons_Iup, input_neurons_Idn, input_neurons_Qup, input_neurons_Qdn]) 
mask = np.zeros(len(network['mRes'].t), dtype=bool)
for n in input_neurons:
    idx = np.where(network['mRes'].i==n)[0]
    mask[idx] = True
num_bins = int(settings['ro_time']/settings['ro_bin_size'])
X = []
for i in range(num_samples):
    time_range = [i*settings['ro_time']/ms, (i+1)*settings['ro_time']/ms]
    ro, _, _ = np.histogram2d(network['mRes'].t[mask]/ms, network['mRes'].i[mask], \
                            bins=[num_bins, params['N']], range=[time_range, [0, params['N']]])
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(ro.T, origin='upper', interpolation='nearest', aspect='auto')
    plt.savefig(exp_dir+"/{}_{}.pdf".format(Y[i][1], Y[i][2]), bbox_inches='tight')
    X.append(ro)

# Store readout and labels
print("- Storing results")
record = {
    'X': X,
    'Y': Y
}
path = exp_dir+"/result.joblib"
with open(path, 'wb') as fo:
    joblib.dump(record, fo)
print("Done!")
    