import time, joblib
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
from utils.dataset import *
from utils.reservoir import getTauCurrent
from brian2 import pA, ms, us, SpikeMonitor, SpikeGeneratorGroup, prefs, device, set_device, defaultclock
from teili import TeiliNetwork
from teili.core.groups import Neurons, Connections
from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn
import sys, os
sys.path.insert(0, '../spiking-radio/runner/components')
from asynchronousdeltamodulator import AsynchronousDeltaModulator
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, fowlkes_mallows_score
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events


def setup_connectivity(N, pIR, pRR):
    """
    Setup the connectivity matrices for the synapsis between
    input layer and reservoir and those within the reservoir itself

    Parameters
    ----------
    N : int
        number of neurons in the reservoir

    pIR : float
        probability of connection for the input neurons
    
    pRR : float
        probability of connection for the reservoir neurons

    Returns
    -------
    connectivity : object
        contains the two connectivity matrices as
        i and j indicesto be used in the connect method
        of the synapse object in Brian2
    """
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

def setup_input_layer(components):
    """
    Setup the input layer consisting of a spike generator with 4 neurons
    that project to 2 input neurons through excitatory and inhibitory synapsis

    Parameters
    ----------
    components : object
        previously defined network components

    Returns
    -------
    components : object
        network components
    """
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

def setup_reservoir_layer(components, connectivity, N, Itau, Wres, Winp):
    """
    Setup the reservoir layer consisting of a group of randomly connected neurons.
    The connections can be excitatory or inhibitory.

    Parameters
    ----------
    components : object
        previously defined network components
    
    connectivity : object
        contains the two connectivity matrices as
        i and j indicesto be used in the connect method
        of the synapse object in Brian2

    N : int
        number of neurons in the reservoir

    Itau : float
        current of the membrane potential decay time of
        the reservoir neurons in pA
    
    Wres : float or ndarray
        weight or weight matrix of the reservoir synapsis

    Winp : float or ndarray
        weight or weight matrix of the input synapsis

    Returns
    -------
    components : object
        network components
    """
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
    sInpRes.weight = Winp
    mRes = SpikeMonitor(gRes, name='mRes')
    components['layers']['gRes'] = gRes
    components['synapsis']['sInpRes'] = sInpRes
    components['synapsis']['sResRes'] = sResRes
    components['monitors']['mRes'] = mRes
    return components

def init_network(indices, times, connectivity, N, Itau):
    """
    Initialize the network with the input stimulus

    Parameters
    ----------
    indices : list
        spike generator neuron indices

    times : list
        spike generator spiking times

    connectivity : object
        contains the two connectivity matrices as
        i and j indicesto be used in the connect method
        of the synapse object in Brian2

    """
    # define fixed reservoir paramenters
    pInh = 0.2
    wInp = 3500
    wRes = 50
    # define reservoir weights
    num_connections = len(connectivity['res_res']['i'])
    Wres = wRes*np.random.choice([-1, 1], size=num_connections, p=[pInh, 1-pInh])
    # setup input layer
    components = {'generator': None, 'layers': None, 'synapsis': None, 'monitors': None}
    components = setup_input_layer(components)
    # setup reservoir layer
    components = setup_reservoir_layer(components, connectivity, N, Itau, Wres, wInp)
    # set spikes
    components['generator'].set_spikes(indices, times)
    # initialize network
    network = TeiliNetwork()
    network.add(components['generator'])
    network.add(list(components['layers'].values()))
    network.add(list(components['synapsis'].values()))
    network.add(list(components['monitors'].values()))
    return network

def readout(monitor, ro_time, num_neurons, tot_num_samples, bin_size=1):
    """
    Parse the spike monitor of the reservoir and extract the activity
    for each sample in the input stimulus by counting the spikes in
    in time intervals of a given size

    Parameters
    ----------
    monitor : SpikeMonitor
        spike monitor of the reservoir
    
    ro_time : float
        width of the time interval from which to read out
        the activity of one sample in ms

    num_neurons : int
        number of neurons in the reservoir

    tot_num_samples : int
        total number of samples in the input stimulus

    bin_size : int
        size of the time bins in ms

    Returns
    -------
    X : list
        activity for each sample

    ro_bins : list
        size of the time and space bins

    edges : list
        time edges from which the activity has been extracted
        for each sample
    """
    X = []
    edges = []
    ro_time = ro_time/ms
    ro_bins = [int(ro_time/bin_size), num_neurons]
    for i in range(tot_num_samples):
        ro_range = [[i*ro_time, (i+1)*ro_time], [0, num_neurons]]
        window = np.logical_and(monitor.t/ms>i*ro_time, monitor.t/ms<=(i+1)*ro_time)
        hist, _, _ = np.histogram2d(monitor.t[window]/ms, monitor.i[window], \
            bins=ro_bins, range=ro_range)
        X.append(hist)
        edges.append(ro_range)
    return X, ro_bins, edges

# Define classifier
def classify(X, Y):
    """
    Classify the activity generated by the reservoir

    Parameters
    ----------
    X : ndarray (num_samples, num_features)
        readout output from the reservoir for each
        sample in the stimulus

    Y : ndarray (num_samples)
        labels for each sample in the stimulus

    Returns
    -------
    score : float
        performance metric of the accuracy of the
        classification (higher is better)
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    try:
        clf = LogisticRegression(random_state=42, solver='lbfgs').fit(X_train, Y_train)
    except Exception as e:
        print(e)
        score = -100.0
    else:
        Y_pred = clf.predict_proba(X_test)
        score = log_loss(Y_test, Y_pred)
    return -1.0*score


def score(X, Y, k):
    """
    Return a score based on the clustering performance of the reservoir

    Parameters
    ----------
    X : ndarray (num_samples, num_features)
        readout output from the reservoir for each sample
        in the stimulus

    Y : ndarray (num_samples)
        labels for each sample in the stimulus

    k : int
        number of clusters (modulation classes)

    Returns
    -------
    score : float
        performance metric of the reservoir (higher is better)
    """
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    Y_pred = kmeans.labels_
    score = fowlkes_mallows_score(Y, Y_pred)
    return score

def plot_raster(monitor, directory):
    """
    Plot the raster of spikes generated by the input stimulus

    Parameters
    ----------
    monitor : SpikeMonitor
        spike monitor of the reservoir

    directory : string
        path to the folder into which the plot should be saved
    """
    # TODO: add pagination for long stimuli
    plt.figure()
    plt.scatter(monitor.t/ms, monitor.i, marker=',', s=2)
    plt.xlabel('time [ms]')
    plt.ylabel('neuron index')
    plt.savefig(directory+'/raster_plot.pdf')

def plot_result(X, Y, bins, edges, modulations, snr, tot_num_samples, directory):
    """
    Plot the readout activity for each sample

    Parameters
    ----------
    X : ndarray (num_samples, num_features)
        readout output from the reservoir for each sample
        in the stimulus

    Y : ndarray (num_samples)
        labels for each sample in the stimulus

    bins : list
        size of the time and space bins

    edges : list
        time edges from which the activity has been extracted
        for each sample

    modulations : list
        modulation classes in the input stimulus

    snr : float
        signal-to-noise level of the input stimulus

    tot_num_samples : int
        total number of samples per modulation class

    directory : string
        path to the folder into which the plot should be saved
    """
    for i in range(tot_num_samples):
        sid = i-Y[i]*num_samples
        plt.figure()
        plt.imshow(X[i].T, interpolation='nearest', origin='low', aspect='auto', \
           extent=[0, bins[0], 0, bins[1]], cmap='viridis')
        plt.title('{} @ {} #{}'.format(modulations[Y[i]], snr, sid))
        plt.xlabel('vector element')
        plt.ylabel('neuron index')
        plt.savefig(directory+'/{}_{}_{}.pdf'.format(modulations[Y[i]], snr, sid))
        
def store_result(X, Y, score, params):
    """
    Store the data produced by the experiment to disk

    Parameters
    ----------
    X : ndarray (num_samples, num_features)
        readout output from the reservoir for each sample
        in the stimulus

    Y : ndarray (num_samples)
        labels for each sample in the stimulus

    score : float
        performance metric of the reservoir (higher is better)

    params : object
        reservoir parameters used in the experiment
    """
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
def experiment(N=50, tau=50, pRR=0.3, \
    indices=None, times=None, stretch_factor=None, duration=None, ro_time=None, \
    modulations=None, snr=None, num_samples=None, Y=None, \
    plot=False, store=False):
    """
    Run an experiment on a reservoir with given properties and evaluate
    its performance in terms of clustering inputs of different classes.

    Parameters
    ----------
    N : int
        number of neurons in the reservoir

    tau : float
        membrane potential decay time in ms of the reservoir
        neurons
    
    pRR : float
        probability of connection for the reservoir neurons

    indices : list
        spike generator neuron indices

    times : list
        spike generator spiking times

    stretch_factor : int
        time steching factor to adapt radioML signals to DPI
        operation time scales

    duration : float
        time length of the concatenated stimulus in ms

    ro_time : float
        width of the time interval from which to read out
        the activity of one sample in ms

    modulations : list
        modulation classes in the input stimulus

    snr : float
        signal-to-noise level of the input stimulus

    num_samples : int
        number of samples per modulation class

    Y : list
        labels of the samples in the input stimulus

    plot : bool
        flag to plot the results from the experiments

    store : bool
        flag to store the results from the experiment

    Returns
    -------
    score : float
        performance metric of the reservoir (higher is better)      
    """
    start = time.perf_counter()
    N = int(N)
    print("- running with: N={}, tau={}, pRR={}".format(N, tau, pRR))
    pIR = 0.3
    connectivity = setup_connectivity(N, pIR, pRR)
    Itau = getTauCurrent(tau*ms)
    # Set C++ backend and time step
    title = 'srres_{}_N{}tau{:.2f}pRR{:.2f}'.format(os.getpid(), N, tau, pRR)
    directory = '../brian2_devices/' + title
    set_device('cpp_standalone', directory=directory, build_on_run=True)
    device.reinit()
    device.activate(directory=directory, build_on_run=True)
    defaultclock.dt = stretch_factor*us
    # Initialize network
    network = init_network(indices, times, connectivity, N, Itau)    
    # Run simulation
    network.run(duration, recompile=True)
    # Readout activity
    tot_num_samples = num_samples*len(modulations)
    X, bins, edges = readout(network['mRes'], ro_time, N, tot_num_samples, bin_size=5)
    if plot:
        plots_dir = directory+'/plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plot_raster(network['mRes'], plots_dir)
        plot_result(X, Y, bins, edges, modulations, snr, tot_num_samples, plots_dir)
    # Measure reservoir perfomance
    X = list(map(lambda x: x.T.flatten(), X))
    s = score(X, Y, len(modulations))
    if store:
        params = {'N': N, 'tau': tau, 'pRR': pRR}
        store_result(X, Y, s, params)
    print("- experiment took {} [s]".format(time.perf_counter()-start))
    return s

if __name__ == '__main__':   
    # TODO: clear devices folder

    # Set brian2 extra compilation arguments
    prefs.devices.cpp_standalone.extra_make_args_unix = ["-j6"]

    # Import dataset and prepare samples
    snr = 18
    # modulations = [
    #     '8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'
    # ]
    modulations = [
        '8PSK', 'BPSK', 'QPSK'
    ]
    num_samples = 20
    tot_num_samples = num_samples*len(modulations)
    dataset = load_dataset('../spiking-radio/data/radioML/RML2016.10a_dict.pkl', snr=snr, normalize=True)
    # Define delta modulators
    time_sample = np.arange(128)
    thrup = 0.01
    thrdn = 0.01
    resampling_factor = 200
    stretch_factor = 50
    modulator = [
        AsynchronousDeltaModulator(thrup, thrdn, resampling_factor),
        AsynchronousDeltaModulator(thrup, thrdn, resampling_factor)
    ]
    # Prepare stimulus
    print("preparing input stimulus")
    indices = []
    times = []
    Y = []
    pause = 5000*ms
    stimulation = (len(time_sample)*resampling_factor*stretch_factor/1e3)*ms
    duration = (stimulation+pause)*num_samples*len(modulations)
    to = 0.0*ms
    for (i, mod) in tqdm(enumerate(modulations)):
        for j in range(num_samples):
            sample = dataset[(mod, snr)][j]
            ix, tx, _, _ = modulate(modulator[0], modulator[1], time_sample, sample, \
                            resampling_factor=resampling_factor, stretch_factor=stretch_factor)
            tx = tx + to
            indices.extend(ix)
            times.extend(tx)
            Y.append(i)
            to = (stimulation+pause)*(i*num_samples+j+1)
    print("\t - duration: {}s".format(duration))

    # Test the experiment function
    # score = experiment(N=50, tau=50, pRR=0.3, \
    #     indices=indices, times=times, stretch_factor=stretch_factor, duration=duration, ro_time=stimulation+pause, \
    #     modulations=modulations, snr=snr, num_samples=num_samples, Y=Y, \
    #     plot=True, store=False)
    # print(score)

    # Define optimization bounds
    pbounds = {
        'N': (50, 200),     # number of neurons   
        'tau': (20, 200),    # DPI time constant [ms]
        'pRR': (0.3, 1.0)   # probability of connection
        # TODO: define good bounds
        #'wRes': ()         # units of baseweight
        #'wInp': ()         # units of baseweight   
    }

    # Define Bayesian optmization process
    def bo_experiment(N=50, tau=50, pRR=0.3):
        return experiment(N=N, tau=tau, pRR=pRR, \
            indices=indices, times=times, stretch_factor=stretch_factor, duration=duration, ro_time=stimulation+pause, \
            modulations=modulations, snr=snr, num_samples=num_samples, Y=Y)

    optimizer = BayesianOptimization(
        f=bo_experiment,
        pbounds=pbounds,
        random_state=42
    )

    # Subscribe logger
    logger = JSONLogger(path="/home/massimo/Development/simulations/srres_bo_logger-{}.json".format(datetime.now().strftime("%Y-%m-%dT%H:%M:%S")))
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    # TODO: try multithreading
    # Optimize model performance
    print("starting optimization")
    start = time.perf_counter()
    optimizer.maximize(
        init_points=2,
        n_iter=25,
    )

    # Print results
    best = optimizer.max
    print("Best solution: ")
    print("\t - score: {}".format(best['target']))
    for (key, value) in best['params'].items():
        print("\t - {}: {}".format(key, value))

    # Print runtime info
    print("optimization took: {}".format(time.perf_counter()-start))