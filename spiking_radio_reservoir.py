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
from sklearn.metrics import log_loss, silhouette_score
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events


def _pAB(a, b, AoC, DoC):
    if a[2]==1 and b[2]==1:
        C = AoC[0]
    elif (a[2]==1 and b[2]==-1) or (a[2]==-1 and b[2]==1):
        C = AoC[1]
    elif a[2]==-1 and b[2]==-1:
        C = AoC[2]
    return C*np.exp(-1*np.linalg.norm(a-b)/DoC**2)

def _wAB(a, b, types):
    if a[2]==-1 and b[2]==1:
        w = -1
    else:
        w = 1
    return w

def setup_connectivity(N, pInh, pIR, Ngx, Ngy, AoC, DoC):
    """
    Setup connectivity matrices for the synapsis between
    input layer and reservoir and those within the reservoir itself
    as folows:
        - the reservoir is split in 2 groups
        - each input neuron projects to one of the groups
        - the neurons in the reservoir are arranged on a 2D-grid
        - the probability of connection between reservoir neurons
          is dependent on the distance between the neurons

    Parameters
    ----------
    N : int
        number of neurons in the reservoir

    pInh : float
        probability that a reservoir neuron is inhibitory

    pIR : float
        probability of connection for the input neurons

    Ngx : int
        number of reservoir neurons in the x-axis of the grid

    Ngy : int
        number of reservoir neurons in the y-axis of the grid

    AoC : list
        list of probability amplitudes for the connections between
        reservoir neurons ([ex-ex, ex-inh, inh-inh])
    
    DoC : float
        density of connections between reservoir neurons

    Returns
    -------
    connectivity : object
        contains the two connectivity matrices as
        i and j indices to be used in the connect method
        of the synapse object in Brian2
    """
    if Ngx*Ngy!=N: 
        raise Exception("Reservoir grid dimensions do not coincide with number of neurons.")
    # organize neurons on grid
    reservoir = np.arange(N)
    grid = reservoir.reshape((Ngx, Ngy))
    # define type of each neuron
    types = np.random.choice([-1, 1], size=N, p=[pInh, 1-pInh])
    # connect reservoir neurons
    Cres = [[], []]
    Wres = []
    for n in reservoir:
        a = np.array(list(zip(*np.where(grid==n)))[0])
        a = np.append(a, types[n])
        for x in range(Ngx):
            for y in range(Ngy):
                b = np.array([x, y, types[x*Ngy+y]])
                p = _pAB(a, b, AoC, DoC)
                r = np.random.uniform()
                if r<p:
                    Cres[0].append(n)
                    Cres[1].append(grid[x, y])
                    Wres.append(_wAB(a, b, types))
    # connect input to reservoir
    Cin = [[], []]
    for n in reservoir:
        r = np.random.uniform()
        if r<pIR:
            Cin[1].append(n)
            if n<int(N/2):
                Cin[0].append(0)
            else:
                Cin[0].append(1)
    # return connectivity and weights
    connectivity = {
        'inp_res': {
            'i': Cin[0],
            'j': Cin[1]
        },
        'res_res': {
            'i': Cres[0],
            'j': Cres[1],
            'w': Wres
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

def setup_reservoir_layer(components, connectivity, N, Itau, wRes, wInp):
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
    
    wRes : float
        weight of the reservoir synapsis

    wInp : float
        weight of the input synapsis

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
    sResRes.weight = wRes*np.array(connectivity['res_res']['w'])
    #sResRes.I_etau = Ietau
    #sResRes.I_itau = Iitau 
    sInpRes = Connections(components['layers']['gInp'], gRes, equation_builder=DPISyn(), method='euler', name='sInpRes')
    sInpRes.connect(i=connectivity['inp_res']['i'], j=connectivity['inp_res']['j'])
    sInpRes.weight = wInp
    mRes = SpikeMonitor(gRes, name='mRes')
    components['layers']['gRes'] = gRes
    components['synapsis']['sInpRes'] = sInpRes
    components['synapsis']['sResRes'] = sResRes
    components['monitors']['mRes'] = mRes
    return components

def init_network(indices, times, connectivity, N, Itau, wRes, wInp):
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

    N : int
        number of neurons in the reservoir

    tau : float
        membrane potential decay time in ms of the reservoir
        neurons

    wInp : float
        weight of the input synapsis
    
    wRes : float
        weight of the reservoir synapsis

    Returns
    -------
    network : TeiliNetwork
        instance of the network with all the commponents
    """
    # setup input layer
    components = {'generator': None, 'layers': None, 'synapsis': None, 'monitors': None}
    components = setup_input_layer(components)
    # setup reservoir layer
    components = setup_reservoir_layer(components, connectivity, N, Itau, wRes, wInp)
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
    score = silhouette_score(X, Y_pred)
    return score

def plot_raster(monitor, directory=None):
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
    fig = plt.figure()
    plt.scatter(monitor.t/ms, monitor.i, marker=',', s=2)
    plt.xlabel('time [ms]')
    plt.ylabel('neuron index')
    if directory:
        plt.savefig(directory+'/raster_plot.pdf')
        plt.close(fig=fig)

def plot_result(X, Y, bins, edges, modulations, snr, tot_num_samples, directory=None):
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
        fig= plt.figure()
        plt.imshow(X[i].T, interpolation='nearest', origin='low', aspect='auto', \
           extent=[0, bins[0], 0, bins[1]], cmap='viridis')
        plt.title('{} @ {} #{}'.format(modulations[Y[i]], snr, sid))
        plt.xlabel('vector element')
        plt.ylabel('neuron index')
        if directory:
            plt.savefig(directory+'/{}_{}_{}.pdf'.format(modulations[Y[i]], snr, sid))
            plt.close(fig=fig)


def plot_network(network, N, weights, directory=None):
    """
    Plot the network layers and connections

    Parameters
    ----------
    network : TeiliNetwork
        instance of the network with all the commponents

    N : int
        number of neurons in the reservoir

    weights : list
        weight of each synaptic connection in the reservoir

    directory : string
        path to the folder into which the plot should be saved
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.scatter(network['sGenInp'].i, network['sGenInp'].j, c='k', marker='.')
    ax1.set_xlabel('Source neuron index')
    ax1.set_ylabel('Target neuron index')
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(['I.up', 'I.dn', 'Q.up', 'Q.dn'])
    ax1.set_yticks([0, 1])
    ax1.set_title('Generator')
    ax2.scatter(network['sInpRes'].i, network['sInpRes'].j, c='k', marker='.')
    ax2.set_xlabel('Source neuron index')
    #ax2.set_ylabel('Target neuron index')
    ax2.set_xticks([0, 1])
    ax2.set_title('Input')
    C = np.zeros((N, N))
    C[network['sResRes'].i, network['sResRes'].j] = weights
    ax3.imshow(C, aspect='auto', origin='lower')
    ax3.set_xlabel('Source neuron index')
    #ax3.set_ylabel('Target neuron index')
    ax3.set_title('Reservoir')
    if directory:
        plt.savefig(directory+'/network_plot.pdf')
        plt.close(fig=fig)

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
def experiment(wInp=3500, wRes=50, DoC=2, 
    N=200, tau=20, Ngx=10, Ngy=20, \
    indices=None, times=None, stretch_factor=None, duration=None, ro_time=None, \
    modulations=None, snr=None, num_samples=None, Y=None, \
    plot=False, store=False):
    """
    Run an experiment on a reservoir with given properties and evaluate
    its performance in terms of clustering inputs of different classes.

    Parameters
    ----------
    wInp : float
        weight of the input synapsis
    
    wRes : float
        weight of the reservoir synapsis

    N : int
        number of neurons in the reservoir

    tau : float
        membrane potential decay time in ms of the reservoir
        neurons

    Ngx : int
        number of reservoir neurons in the x-axis of the grid

    Ngy : int
        number of reservoir neurons in the y-axis of the grid

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
    print("- running with: wInp={}, wRes={}, DoC={}".format(wRes, wInp, DoC))
    pIR = 0.3
    pInh = 0.2
    AoC = [0.3, 0.5, 0.1]
    connectivity = setup_connectivity(N, pInh, pIR, Ngx, Ngy, AoC, DoC)
    Itau = getTauCurrent(tau*ms)
    # Set C++ backend and time step
    title = 'srres_{}_wInp{}wRes{:.2f}DoC{:.2f}'.format(os.getpid(), wInp, wRes, DoC)
    directory = '../brian2_devices/' + title
    set_device('cpp_standalone', directory=directory, build_on_run=True)
    device.reinit()
    device.activate(directory=directory, build_on_run=True)
    defaultclock.dt = stretch_factor*us
    # Initialize network
    network = init_network(indices, times, connectivity, N, Itau, wRes, wInp)    
    # Run simulation
    network.run(1000*ms, recompile=True)
    # Readout activity
    tot_num_samples = num_samples*len(modulations)
    X, bins, edges = readout(network['mRes'], ro_time, N, tot_num_samples, bin_size=5)
    if plot:
        plots_dir = directory+'/plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plot_raster(network['mRes'], plots_dir)
        plot_result(X, Y, bins, edges, modulations, snr, tot_num_samples, plots_dir)
        plot_network(network, N, connectivity['res_res']['w'], plots_dir)
    # Measure reservoir perfomance
    X = list(map(lambda x: x.T.flatten(), X))
    s = score(X, Y, len(modulations))
    if store:
        params = {'wInp': wInp, 'wRes': wRes, 'DoC': DoC}
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
    score = experiment(wInp=3500, wRes=100, DoC=4,
        N=200, tau=20, Ngx=10, Ngy=20, \
        indices=indices, times=times, stretch_factor=stretch_factor, duration=duration, ro_time=stimulation+pause, \
        modulations=modulations, snr=snr, num_samples=num_samples, Y=Y, \
        plot=True, store=False)
    print(score)

    # Define optimization bounds
    # pbounds = {
    #     #'N': (50, 200),     # number of neurons   
    #     #'tau': (20, 200),    # DPI time constant [ms]
    #     #'pRR': (0.3, 1.0)   # probability of connection
    #     # TODO: define good bounds
    #     'wRes': (),         # units of baseweight
    #     'wInp': (),         # units of baseweight
    #     'DoC': ()           # density of connection   
    # }

    # # Define Bayesian optmization process
    # def bo_experiment(wInp=3500, wRes=50, DoC=2):
    #     return experiment(wInp=wInp, wRes=wRes, DoC=DoC,
    #         N=N, tau=tau, \
    #         indices=indices, times=times, stretch_factor=stretch_factor, duration=duration, ro_time=stimulation+pause, \
    #         modulations=modulations, snr=snr, num_samples=num_samples, Y=Y)

    # optimizer = BayesianOptimization(
    #     f=bo_experiment,
    #     pbounds=pbounds,
    #     random_state=42
    # )

    # # Subscribe logger
    # logger = JSONLogger(path="/home/massimo/Development/simulations/srres_bo_logger-{}.json".format(datetime.now().strftime("%Y-%m-%dT%H:%M:%S")))
    # optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    # # TODO: try multithreading
    # # Optimize model performance
    # print("starting optimization")
    # start = time.perf_counter()
    # optimizer.maximize(
    #     init_points=2,
    #     n_iter=25,
    # )

    # # Print results
    # best = optimizer.max
    # print("Best solution: ")
    # print("\t - score: {}".format(best['target']))
    # for (key, value) in best['params'].items():
    #     print("\t - {}: {}".format(key, value))

    # # Print runtime info
    # print("optimization took: {}".format(time.perf_counter()-start))