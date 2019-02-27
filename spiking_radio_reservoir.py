import time, joblib
import numpy as np
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
from utils.dataset import *
from utils.reservoir import getTauCurrent
from brian2 import pA, ms, SpikeMonitor, SpikeGeneratorGroup, prefs
from teili import TeiliNetwork
from teili.core.groups import Neurons, Connections
from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn
import sys
sys.path.insert(0, '../spiking-radio/runner/components')
from asynchronousdeltamodulator import AsynchronousDeltaModulator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

# Define connectivity
def setup_connectivity(N, pIR, pRR):
    Cin = np.random.choice([0, 1], size=(2, N), p=[1-pIR, pIR]).nonzero()
    Cres = np.random.choice([0, 1], size=(N, N), p=[1-pRR, pRR]).nonzero()
    connectivity = {
        'inp_res': {
            'i': Cin[0],
            'j': Cin[1]
        },
        'res_res': {
            'i': Cres[0],
            'j': Cres[1]
        }
    }
    return connectivity

# Define input layer
def setup_input_layer(components):
    gGen = SpikeGeneratorGroup(4, np.array([]), np.array([])*ms, name='gGen')
    gInp = Neurons(2, equation_builder=DPI(num_inputs=2), refractory=0.0*ms, name='gInp')
    gInp.Iahp = 0.5*pA
    sGenInp = Connections(gGen, gInp, equation_builder=DPISyn(), method='euler', name='sGenInp')
    sGenInp.connect(i=[0, 1, 2, 3], j=[0, 0, 1, 1])
    sGenInp.weight = [3500, -3500, 3500, -3500]
    mGen = SpikeMonitor(gGen, name='mGen')
    mInp = SpikeMonitor(gInp, name='mInp')
    components['generator'] = gGen
    components['layers'] = {'gInp': gInp}
    components['synapsis'] = {'sGenInp': sGenInp}
    components['monitors'] = {'mGen': mGen, 'mInp': mInp}
    return components

# Define reservoir layer
def setup_reservoir_layer(components, connectivity, N, Itau, Wres, Win):
    gRes = Neurons(N, equation_builder=DPI(num_inputs=2), refractory=0.0*ms, name='gRes')
    gRes.Iahp = 0.5*pA
    gRes.Itau = Itau
    sResRes = Connections(gRes, gRes, equation_builder=DPISyn(), method='euler', name='sResRes')
    sResRes.connect(i=connectivity['res_res']['i'], j=connectivity['res_res']['j'])
    sResRes.weight = Wres
    #sResRes.I_etau = Ietau
    #sResRes.I_itau = Iitau 
    sInpRes = Connections(components['layers']['gInp'], gRes, equation_builder=DPISyn(), method='euler', name='sInpRes')
    sInpRes.connect(i=connectivity['inp_res']['i'], j=connectivity['inp_res']['j'])
    sInpRes.weight = Win
    mRes = SpikeMonitor(gRes, name='mRes')
    components['layers']['gRes'] = gRes
    components['synapsis']['sInpRes'] = sInpRes
    components['synapsis']['sResRes'] = sResRes
    components['monitors']['mRes'] = mRes
    return components

# Define reservoir
def reservoir(connectivity, N, Itau, Wres, Win):
    # setup input layer
    components = {'generator': None, 'layers': None, 'synapsis': None, 'monitors': None}
    components = setup_input_layer(components)
    # setup reservoir layer
    components = setup_reservoir_layer(components, connectivity, N, Itau, Wres, Win)
    # initialize network
    network = TeiliNetwork()
    network.add(components['generator'])
    network.add(list(components['layers'].values()))
    network.add(list(components['synapsis'].values()))
    network.add(list(components['monitors'].values()))
    network.store('initialized')
    return network

# Define readout
def readout(monitor):
    hist, _, _ = np.histogram2d(monitor.t/ms, monitor.i, bins=ro_bins, range=ro_range)
    return hist.T.flatten()

# Define classifier
def classify(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    try:
        clf = LogisticRegression(random_state=42, solver='lbfgs').fit(X_train, Y_train)
    except Exception as e:
        score = 0.0
    else:
        Y_pred = clf.predict_proba(X_test)
        score = log_loss(Y_test, Y_pred)
    return -1.0*score

# Define initialization of network
def init_network(*args):
    current = mp.current_process()
    print("{}: initializing".format(current.name))
    connectivity = args[0]
    N = args[1]
    Itau = args[2]
    # define global variables
    global modulator
    global duration
    global time_sample
    global resampling_factor
    global stretch_factor
    global bin_size
    global network
    global ro_bins
    global ro_range
    # define delta modulators
    time_sample = np.arange(128)
    thrup = 0.01
    thrdn = 0.01
    resampling_factor = 200
    stretch_factor = 50
    modulator = [
        AsynchronousDeltaModulator(thrup, thrdn, resampling_factor),
        AsynchronousDeltaModulator(thrup, thrdn, resampling_factor)
    ]
    duration = len(time_sample) * resampling_factor * stretch_factor / 1e3
    # define fixed reservoir paramenters
    pInh = 0.2
    wInp = 3500
    wRes = 50
    # define network
    num_connections = len(connectivity['res_res']['i'])
    Wres = wRes*np.random.choice([-1, 1], size=num_connections, p=[pInh, 1-pInh])
    network = reservoir(connectivity, N, Itau, Wres, wInp)
    # define readout paramenters
    bin_size = 5
    ro_bins = [int(duration/bin_size), N]
    ro_range = [[0, duration], [0, N]]

# Define stimulation of network
def stimulate_network(x):
    # current = mp.current_process()
    # print("{}: stimulating with sample {} of class {}".format(current.name, x[2], x[1]))
    indices, times, _, _ = modulate(modulator[0], modulator[1], time_sample, x[0], \
                            resampling_factor=resampling_factor, stretch_factor=stretch_factor)
    network.restore('initialized')
    network['gGen'].set_spikes(indices, times)
    try:
        network.run(duration*ms)
    except Exception as e:
        print("Run error: ", e)
    activity = readout(network['mRes'])
    return [activity, x[1]]

def store_result(X, Y, score, params):
    record = {
        'X': X,
        'Y': Y,
        'score': score,
        'params': params
    }
    path = './results/{}.joblib'.format(datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    with open(path, 'wb') as fo:
        joblib.dump(record, fo)

# Define experiment
def experiment(N, tau, pRR, store=False):
    N = int(N)
    # print("- running with: N={}, tau={}, pRR={}".format(N, tau, pRR))
    pIR = 0.3
    connectivity = setup_connectivity(N, pIR, pRR)
    Itau = getTauCurrent(tau*ms)    
    # initialize pool of processes
    pool = mp.Pool(4, initializer=init_network, initargs=(connectivity, N, Itau))
    result = list(tqdm(pool.imap(stimulate_network, samples), total=len(samples)))
    # classify activity and measure accuracy
    X = list(map(lambda x: x[0], result))
    Y = list(map(lambda x: x[1], result))
    score = classify(X, Y)
    if store:
        params = {'N': N, 'tau': tau, 'pRR': pRR}
        store_result(X, Y, score, params)
    return score

if __name__ == '__main__':
    # Set C++ backend
    # set_device('cpp_standalone', directory='Brian2Network_standalone', build_on_run=True)
    
    # Set numpy code generation
    prefs.codegen.target = 'numpy'

    # Import dataset and prepare samples
    snr = 18
    # modulations = [
    #     '8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'
    # ]
    modulations = [
        '8PSK', 'BPSK', 'QPSK'
    ]
    num_samples = 100
    dataset = load_dataset('../spiking-radio/data/radioML/RML2016.10a_dict.pkl', snr=snr, normalize=True)
    samples = []
    for (i, mod) in enumerate(modulations):
        for j in range(num_samples):
            sample = dataset[(mod, snr)][j]
            samples.append([sample, i, j])

    # Define optimization bounds
    pbounds = {
        'N': (50, 300),     # number of neurons   
        'tau': (5, 500),    # DPI time constant [ms]
        'pRR': (0.3, 1.0)   # probability of connection
        # TODO: define good bounds
        #'wRes': ()         # units of baseweight
        #'wInp': ()         # units of baseweight   
    }

    # Define Bayesian optmization process
    optimizer = BayesianOptimization(
        f=experiment,
        pbounds=pbounds,
        random_state=42
    )

    # Subscribe logger
    logger = JSONLogger(path="/home/massimo/Development/simulations/srres_bo_logger-{}.json".format(datetime.now().strftime("%Y-%m-%dT%H:%M:%S")))
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    # Optimize model performance
    print("starting optimization")
    start = time.perf_counter()
    optimizer.maximize(
        init_points=2,
        n_iter=50,
    )

    # Print results
    best = optimizer.max
    print("Best solution: ")
    print("\t - score: {}".format(best['target']))
    for (key, value) in best['params']:
        print("\t - {}: {}".format(key, value))

    # Print runtime info
    print("optimization took: {}".format(time.perf_counter()-start))