import time, joblib, warnings
import os, shutil
import numpy as np
from datetime import datetime
from tqdm import tqdm
from brian2 import nA, pA, amp, ms, us, SpikeMonitor, StateMonitor, SpikeGeneratorGroup, prefs, device, set_device, defaultclock
from teili import TeiliNetwork
from teili.core.groups import Neurons, Connections
from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from utils.plotting import *


def _schliebs_pAB(a, b, AoC, DoC):
    if a[3]==1 and b[3]==1:
        C = AoC[0]
    elif a[3]==1 and b[3]==-1:
        C = AoC[1]
    elif a[3]==-1 and b[3]==1:
        C = AoC[2]
    elif a[3]==-1 and b[3]==-1:
        C = AoC[3]
    return C*np.exp(-0.5*(np.linalg.norm(a[:3]-b[:3])/DoC)**2)

def _schliebs_coords(n, grid):
    x = np.zeros(4, dtype=np.float)
    c = np.array(list(zip(*np.where(grid[:,:,:,0]==n)))[0])
    for (i, dim) in enumerate(grid.shape[:-1]):
        x[i] = c[i]/dim
    x[-1] = grid[c[0], c[1], c[2], 1]
    return x

def _scliebs_norm(a, neurons, AoC, DoC, grid):
    norm = 0
    for n in neurons:
        b = _schliebs_coords(n, grid)
        norm += _schliebs_pAB(a, b, AoC, DoC)
    return norm

def _schliebs_wAB(a, b, loc_wResE, scale_wResE, loc_wResI, scale_wResI):
    if a[3]==-1 and b[3]==1:
        w = 1
        while w>0:
            w = np.random.normal(loc_wResI, scale_wResI)
    else:
        w = -1
        while w<0:
            w = np.random.normal(loc_wResE, scale_wResE)
    return w

def setup_schliebs_connectivity(N, pInh, pIR, Ngx, Ngy, Ngz, AoC, DoC, \
    loc_wResE, scale_wResE, loc_wResI, scale_wResI, rebalance=False):
    """
    Setup connectivity matrices for the synapses between
    input layer and reservoir and those within the reservoir itself
    as folows:
        - the reservoir is split in 2 groups
        - each input neuron projects to one of the groups
        - the neurons in the reservoir are arranged on a 3D-grid
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

    Ngz : int
        number of reservoir neurons in the z-axis of the grid

    AoC : list
        list of probability amplitudes for the connections between
        reservoir neurons ([ex-ex, ex-inh, inh-ex, inh-inh])
    
    DoC : float
        density of connections between reservoir neurons

    loc_wResE : float (positive)
        mean value of the reservoir excitatory weights distribution

    scale_wResE : float
        standard deviation of the reservoir excitatory weights distribution

    loc_wResI : float (negative)
        mean value of the reservoir inhibitory weights distribution

    scale_wResI : float
        standard deviation of the reservoir inhibitory weights distribution

    rebalance : bool
        wheter the reservoir weights should be adjusted to satisfy the
        input balance condition for each neuron

    Returns
    -------
    connectivity : dict
        contains the two connectivity matrices as
        i and j indices to be used in the connect method
        of the synapse object in Brian2

    Notes
    -----
    The method is inspired by the paper:
    "Reservoir-Based Evolving Spiking Neural Network
    for Spatio-temporal Pattern Recognition" - Schliebs et al. 2011
    """
    if Ngx*Ngy*Ngz!=N: 
        raise Exception("Reservoir grid dimensions do not coincide with number of neurons.")
    # set reservoir neurons and define type of each
    neurons = np.arange(N)    
    types = np.random.choice([-1, 1], size=N, p=[pInh, 1-pInh])
    reservoir = np.array(list(zip(neurons, types)))
    # organize neurons on grid    
    grid = reservoir.reshape((Ngx, Ngy, Ngz, -1))
    # connect reservoir neurons
    Cres = [[], []]
    Wres = []
    for n in neurons:
        a = _schliebs_coords(n, grid)
        norm = 1.0 #_norm(a, AoC, DoC, grid)
        for m in neurons:
            b = _schliebs_coords(m, grid)
            p = _schliebs_pAB(a, b, AoC, DoC)/norm
            r = np.random.uniform()
            if r<p:
                Cres[0].append(n)
                Cres[1].append(m)
                Wres.append(_schliebs_wAB(a, b, loc_wResE, scale_wResE, loc_wResI, scale_wResI))
    Wres = np.array(Wres)
    # rebalance weights if needed
    if rebalance:
        epsilon = 1e-3
        Wres_new = np.zeros((N, N))
        Wres_new[Cres[0], Cres[1]] = Wres
        for n in neurons:
            I = np.where(Wres_new[:, n]<0)[0]
            if len(I)>0:
                delta = -np.sum(Wres_new[:, n])/len(I)
                Wres_new[I, n] += delta+epsilon
            E = np.where(Wres_new[:, n]>0)[0]
            if len(E)>0:
                delta = -np.sum(Wres_new[:, n])/len(E)
                Wres_new[E, n] += delta+epsilon
        if len(Wres_new.nonzero()[0]) != len(Wres.nonzero()[0]):
            raise Exception("Rebalancing error.")
        else:
            Wres = Wres_new[Wres_new.nonzero()].flatten() 
    # connect input to reservoir
    Cin = [[], []]
    for m in range(4):
        for n in neurons:
            r = np.random.uniform()
            if r<pIR:
                Cin[0].append(m)
                Cin[1].append(n)
    # connect generator to input
    Cgen = [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4], [0, 1, 1, 0, 2, 3, 3, 2, 0, 1, 2, 3]]
    Wgen = [1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1]
    # return connectivity and weights
    connectivity = {
        'gen_inp': {
            'i': np.array(Cgen[0]),
            'j': np.array(Cgen[1]),
            'w': np.array(Wgen)
        },
        'inp_res': {
            'i': np.array(Cin[0]),
            'j': np.array(Cin[1])
        },
        'res_res': {
            'i': np.array(Cres[0]),
            'j': np.array(Cres[1]),
            'w': np.array(Wres)
        },
        'grid': grid,
        'types': types
    }
    return connectivity

def _hennequin_pAB(a, b, Ngx, Ngy, p_local, k, DoC):
    l = list(zip(np.random.choice(np.arange(Ngx)/Ngx, size=3), np.random.choice(np.arange(Ngy)/Ngy, size=3)))
    long_range_sum = 0
    for i in range(k):
        long_range_sum += np.exp(-0.5*(np.linalg.norm(a-l[i])/DoC)**2)
    p = p_local*np.exp(-0.5*(np.linalg.norm(a-b)/DoC)**2)+(1-p_local)*long_range_sum/k
    return p

def _hennequin_coords(n, grid):
    x = np.zeros(2, dtype=np.float)
    c = np.array(list(zip(*np.where(grid==n)))[0])
    for (i, dim) in enumerate(grid.shape[1:]):
        x[i] = c[i+1]/dim
    return x

def _hennequin_norm(a, neurons, DoC, grid):
    norm = 0
    for n in neurons:
        b = _hennequin_coords(n, grid)
        norm += np.exp(-0.5*(np.linalg.norm(a-b)/DoC)**2)
    return norm

def _hennequin_wAB(ta, loc_wResE, scale_wResE, loc_wResI, scale_wResI):
    if ta==-1:
        w = 1
        while w>0:
            w = np.random.normal(loc_wResI, scale_wResI)
    else:
        w = -1
        while w<0:
            w = np.random.normal(loc_wResE, scale_wResE)
    return w

def setup_hennequin_connectivity(N, pIR, Ngx, Ngy, pE_local, pI_local, k, DoC, \
    loc_wResE, scale_wResE, loc_wResI, scale_wResI):
    """
    Setup connectivity matrices for the synapses between
    input layer and reservoir and those within the reservoir itself
    as folows:
        - the reservoir is split in 2 2D-grids of size (Ngx, Ngy):
          one for excitatory neurons and one for inhibitory
        - each input neuron projects to a subset of reservoir neurons
        - both excitatory and inhibitory neurons can connect locally
          and at long-range with a distance depedent probability

    Parameters
    ----------
    N : int
        number of neurons in the reservoir

    pIR : float
        probability of connection for the input neurons

    Ngx : int
        number of reservoir neurons in the x-axis of the grid

    Ngy : int
        number of reservoir neurons in the y-axis of the grid

    pE_local : float
        fraction of outgoing synapses that project onto neurons
        in a local neighborhood from excitatory neurons

    pI_local : float
        fraction of outgoing synapses that project onto neurons
        in a local neighborhood from inhibitory neurons

    k : int
        number of long-range synapses
    
    DoC : float
        density of connections between reservoir neurons

    loc_wResE : float (positive)
        mean value of the reservoir excitatory weights distribution

    scale_wResE : float
        standard deviation of the reservoir excitatory weights distribution

    loc_wResI : float (negative)
        mean value of the reservoir inhibitory weights distribution

    scale_wResI : float
        standard deviation of the reservoir inhibitory weights distribution

    Returns
    -------
    connectivity : dict
        contains the two connectivity matrices as
        i and j indices to be used in the connect method
        of the synapse object in Brian2

    Notes
    -----
    The method is inspired by the paper:
    "Stability in spatially structured networks via local inhibition"
    Hennequin et al. 2013
    """
    if 2*Ngx*Ngy!=N: 
        raise Exception("Reservoir grid dimensions do not coincide with number of neurons.")
    M = int(Ngx*Ngy)
    # se treservoir neurons and organize them on a grid
    neurons = np.arange(N)
    types = np.array([1 for i in range(M)] + [-1 for i in range(M)])
    grid = neurons.reshape((2, Ngx, Ngy))
    # connect reservoir neurons
    Cres = [[], []]
    Wres = []
    for n in neurons:
        a = _hennequin_coords(n, grid)
        if n<M:
            p_local = pE_local
        else:
            p_local = pI_local
        norm = 1.0 #_hennequin_norm(a, neurons, DoC, grid)
        for m in neurons:
            b = _hennequin_coords(m, grid)
            p = _hennequin_pAB(a, b, Ngx, Ngy, p_local, k, DoC)/norm
            r = np.random.uniform()
            if r<p:
                Cres[0].append(n)
                Cres[1].append(m)
                Wres.append(_hennequin_wAB(types[n], loc_wResE, scale_wResE, loc_wResI, scale_wResI))
    # connect input to reservoir
    Cin = [[], []]
    I = int(pIR*M)
    input_neurons_exc = np.random.choice(np.arange(M), size=I, replace=False)
    input_neurons_inh = np.random.choice(np.arange(M, 2*M), size=I, replace=False)
    J = int(I/2)
    for (k, m) in enumerate([0, 2]):
        for n in input_neurons_exc[k*J:(k+1)*J]:
            Cin[0].append(m)
            Cin[1].append(n)
    for (k, m) in enumerate([1, 3]):
        for n in input_neurons_inh[k*J:(k+1)*J]:
            Cin[0].append(m)
            Cin[1].append(n)
    # connect generator to input
    Cgen = [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4], [0, 1, 1, 0, 2, 3, 3, 2, 0, 1, 2, 3]]
    Wgen = [1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1]
    # return connectivity and weights
    connectivity = {
        'gen_inp': {
            'i': np.array(Cgen[0]),
            'j': np.array(Cgen[1]),
            'w': np.array(Wgen)
        },
        'inp_res': {
            'i': np.array(Cin[0]),
            'j': np.array(Cin[1])
        },
        'res_res': {
            'i': np.array(Cres[0]),
            'j': np.array(Cres[1]),
            'w': np.array(Wres)
        },
        'grid': grid,
        'types': types
    }
    return connectivity

def setup_generator(components):
    """
    Setup the spike generator with 4+1 neurons and a spike monitor.
    The first 4 neurons represent the 4 input components in the
    following order: [I.up, I.dn, Q.up, Q.dn]. The last neuron
    fires a stop signal at the end of each input stimulus
    
    Parameters
    ----------
    components : object
        previously defined network components

    Returns
    -------
    components : dict
        network components
    """
    gGen = SpikeGeneratorGroup(5, np.array([]), np.array([])*ms, name='gGen')
    mGen = SpikeMonitor(gGen, name='mGen')
    components['generator'] = gGen
    components['monitors']['mGen'] = mGen
    return components

def setup_input_layer(components, connectivity, mismatch, Ninp, currents, wGen):
    """
    Setup the input layer consisting of a spike generator with 4 neurons
    that project to 2 input neurons through excitatory and inhibitory synapses

    Parameters
    ----------
    components : object
        previously defined network components

    connectivity : object
        contains the two connectivity matrices as
        i and j indices to be used in the connect method
        of the synapse object in Brian2

    mismatch : dict
        dictionary with percentual standard deviation of mismatch 
        to add to reservoir neurons and synapses properties

    Ninp : int
        number of input neurons

    currents : dict
        dictionary with values of different currents for the input and
        reservoir neurons and synapses

    wGen : float
        weight of the generator synapses

    Returns
    -------
    components : dict
        network components
    """
    gInp = Neurons(Ninp, equation_builder=DPI(num_inputs=2), refractory=0.0*ms, name='gInp')
    if 'gInp' in mismatch.keys():
        gInp.add_mismatch(mismatch['gInp'], seed=42)
    if 'gInp' in currents.keys():
        for (key, value) in currents['gInp'].items():
            setattr(gInp, key, value)
    sGenInp = Connections(components['generator'], gInp, equation_builder=DPISyn(), method='euler', name='sGenInp')
    sGenInp.connect(i=connectivity['gen_inp']['i'], j=connectivity['gen_inp']['j'])
    sGenInp.weight = wGen*connectivity['gen_inp']['w']
    mInp = SpikeMonitor(gInp, name='mInp')
    smInp = StateMonitor(gInp, ['Imem'], name='smInp', record=True)
    components['layers']['gInp'] = gInp
    components['synapses']['sGenInp'] = sGenInp
    components['monitors']['mInp'] = mInp
    components['monitors']['smInp'] = smInp
    return components

def setup_reservoir_layer(components, connectivity, mismatch, N, currents, wInp, direct_input=False):
    """
    Setup the reservoir layer consisting of a group of randomly connected neurons.
    The connections can be excitatory or inhibitory.

    Parameters
    ----------
    components : object
        previously defined network components
    
    connectivity : object
        contains the two connectivity matrices as
        i and j indices to be used in the connect method
        of the synapse object in Brian2

    mismatch : dict
        dictionary with percentual standard deviation of mismatch 
        to add to reservoir neurons and synapses properties 

    N : int
        number of neurons in the reservoir

    currents : dict
        dictionary with values of different currents for the input and
        reservoir neurons and synapses

    wInp : float
        weight of the input synapses

    direct_input: bool
        wheter the input to the reservoir should be coming from
        the generators directly or not

    Returns
    -------
    components : dict
        network components
    """
    gRes = Neurons(N, equation_builder=DPI(num_inputs=2), refractory=0.0*ms, name='gRes')
    if 'gRes' in currents.keys():
        for (key, value) in currents['gRes'].items():
            setattr(gRes, key, value)
    if 'gRes' in mismatch.keys():
        gRes.add_mismatch(mismatch['gRes'], seed=42)
    if direct_input:
        sInpRes = Connections(components['generator'], gRes, equation_builder=DPISyn(), method='euler', name='sInpRes')
    else:
        sInpRes = Connections(components['layers']['gInp'], gRes, equation_builder=DPISyn(), method='euler', name='sInpRes')
    sInpRes.connect(i=connectivity['inp_res']['i'], j=connectivity['inp_res']['j'])
    sInpRes.weight = wInp
    if 'sInpRes' in currents.keys():
        for (key, value) in currents['sInpRes'].items():
            setattr(sInpRes, key, value)
    sResRes = Connections(gRes, gRes, equation_builder=DPISyn(), method='euler', name='sResRes')
    sResRes.connect(i=connectivity['res_res']['i'], j=connectivity['res_res']['j'])
    sResRes.weight = connectivity['res_res']['w']
    if 'sResRes' in currents.keys():
        for (key, value) in currents['sResRes'].items():
            setattr(sResRes, key, value)
    if 'sResRes' in mismatch.keys():
        sResRes.add_mismatch(mismatch['sResRes'], seed=42)
    mRes = SpikeMonitor(gRes, name='mRes')
    recorded_synapsis = np.random.randint(0, high=len(connectivity['res_res']['i']), size=10)
    smRes = StateMonitor(sResRes, ['Ie_syn', 'Ii_syn'], name='smRes', record=recorded_synapsis)
    components['layers']['gRes'] = gRes
    components['synapses']['sInpRes'] = sInpRes
    components['synapses']['sResRes'] = sResRes
    components['monitors']['mRes'] = mRes
    components['monitors']['smRes'] = smRes
    return components

def init_network(components, indices, times):
    """
    Initialize the network with the input stimulus

    Parameters
    ----------
    components : dict
        network components

    indices : list
        spike generator neuron indices

    times : list
        spike generator spiking times

    connectivity : object
        contains the two connectivity matrices as
        i and j indices to be used in the connect method
        of the synapse object in Brian2

    Returns
    -------
    network : TeiliNetwork
        instance of the network with all the commponents
    """
    components['generator'].set_spikes(indices, times)
    network = TeiliNetwork()
    network.add(components['generator'])
    network.add(list(components['layers'].values()))
    network.add(list(components['synapses'].values()))
    network.add(list(components['monitors'].values()))
    return network

def readout(network, connectivity, ro_time, num_neurons, tot_num_samples, bin_size=1):
    """
    Parse the spike monitor of the reservoir and extract the activity
    for each sample in the input stimulus by counting the spikes in
    in time intervals of a given size

    Parameters
    ----------
    network : TeiliNetwork
        instance of the network with all the commponents

    connectivity : object
        contains the two connectivity matrices as
        i and j indices to be used in the connect method
        of the synapse object in Brian2
    
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
    input_neurons = []
    for i in np.unique(connectivity['inp_res']['i']):
        input_neurons.extend(connectivity['inp_res']['j'][connectivity['inp_res']['i']==i])
    mask = np.zeros(len(network['mRes'].t), dtype=bool)
    for n in input_neurons:
        idx = np.where(network['mRes'].i==n)[0]
        mask[idx] = True
    X = []
    edges = []
    ro_time = ro_time/ms
    ro_bins = [int(ro_time/bin_size), num_neurons]
    for i in range(tot_num_samples):
        ro_range = [[i*ro_time, (i+1)*ro_time], [0, num_neurons]]
        hist, _, _ = np.histogram2d(network['mRes'].t[mask]/ms, network['mRes'].i[mask], \
            bins=ro_bins, range=ro_range)
        X.append(hist)
        edges.append(ro_range)
    return np.array(X), ro_bins, edges

# Define classifier
def classify(X, Y):
    """
    Classify the activity generated by the reservoir

    Parameters
    ----------
    X : list (num_samples)
        activity for each sample

    Y : ndarray (num_samples)
        labels for each sample in the stimulus

    Returns
    -------
    accuracy : list
        classification accuracy at every time bin
    """
    k = X[0].shape[0]
    classes = np.unique(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    accuracy = np.zeros(k)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        for i in tqdm(range(k)):
            samples_train = []
            labels_train = []
            samples_test = []
            labels_test = []
            for c in classes:
                idx_train = np.where(Y_train==c)[0]
                for x in X_train[idx_train]:
                    samples_train.append(x[i])
                    labels_train.append(c)
                idx_test = np.where(Y_test==c)[0]
                for x in X_test[idx_test]:
                    samples_test.append(x[i])
                    labels_test.append(c)
            clf = LogisticRegressionCV(cv=5, multi_class='multinomial')
            try:
                clf.fit(samples_train, labels_train)
            except ConvergenceWarning:
                pass
            else:
                score = clf.score(samples_test, labels_test)
                accuracy[i] = score
    return accuracy


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
    try:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        Y_pred = kmeans.labels_
        score = silhouette_score(X, Y_pred)
    except:
        score = -1.0
    return score

def store_result(X, Y, accuracy, params, title='result', directory=None):
    """
    Store the data produced by the experiment to disk

    Parameters
    ----------
    X : ndarray (num_samples, num_features)
        readout output from the reservoir for each sample
        in the stimulus

    Y : ndarray (num_samples)
        labels for each sample in the stimulus

    accuracy : list
        classification accuracy at every time bin

    params : object
        reservoir parameters used in the experiment

    directory : path
        path where to store the result
    """
    record = {
        'X': X,
        'Y': Y,
        'accuracy': accuracy,
        'params': params
    }
    if directory is not None:
        path = '{}/{}.joblib'.format(directory, title)
    else:
        path = './{}.joblib'.format(title)
    with open(path, 'wb') as fo:
        joblib.dump(record, fo)

# Define experiment
def experiment(wGen=3500, wInp=3500, connectivity=None, mismatch=None, \
    N=200, Ninp=4, currents=None, Ngx=5, Ngy=5, Ngz=8, direct_input=False, \
    indices=None, times=None, stretch_factor=None, duration=None, ro_time=None, \
    modulations=None, snr=None, num_samples=None, Y=None, \
    plot=False, store=False, title=None, exp_dir=None, dt=100*us, remove_device=False):
    """
    Run an experiment on a reservoir with given properties and evaluate
    its performance in terms of clustering inputs of different classes.

    Parameters
    ----------
    wGen : float
        weight of the generator synapses

    wInp : float
        weight of the input synapses

    connectivity : dict
        contains the two connectivity matrices as
        i and j indices to be used in the connect method
        of the synapse object in Brian2

    N : int
        number of neurons in the reservoir
        
    Ninp : int
        number of input neurons

    currents : dict
        dictionary with values of different currents for the input and
        reservoir neurons and synapses

    Ngx : int
        number of reservoir neurons in the x-axis of the grid

    Ngy : int
        number of reservoir neurons in the y-axis of the grid

    Ngz : int
        number of reservoir neurons in the z-axis of the grid

    direct_input: bool
        wheter the input to the reservoir should be coming from
        the generators directly or not

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

    plot : object
        object of key/value pairs to plot different
        aspects of the experiment

    store : bool
        flag to store the results from the experiment

    title : string
        unique title for the experiment

    exp_dir : path
        path to the experiment directory where to store
        the plots

    dt : float (us)
        time step of the simulation in us

    remove_device : bool
        wheter the device folder should be removed after
        the experiment or not

    Returns
    -------
    score : float
        performance metric of the reservoir (higher is better)      
    """
    params = dict(locals())
    start = time.perf_counter()
    # Set C++ backend and time step    
    if title is None:
        title = 'srres_{}'.format(os.getpid())
    directory = '../brian2_devices/' + title
    set_device('cpp_standalone', directory=directory, build_on_run=True)
    device.reinit()
    device.activate(directory=directory, build_on_run=True)
    defaultclock.dt = dt
    # Setup network components
    components = {'generator': None, 'layers': {}, 'synapses': {}, 'monitors': {}}
    components = setup_generator(components)
    if not direct_input:
        components = setup_input_layer(components, connectivity, mismatch, Ninp, currents, wGen)
    components = setup_reservoir_layer(components, connectivity, mismatch, N, currents, wInp, direct_input)
    # Define neurons and synapses reset function
    components['layers']['run_reg_gRes'] = components['layers']['gRes'].run_regularly("Imem=0*pA", dt=ro_time)
    components['synapses']['run_reg_sResRes'] = components['synapses']['sResRes'].run_regularly("""
            Ie_syn=Io_syn
            Ii_syn=Io_syn
        """, dt=ro_time)
    # Initialize network
    network = init_network(components, indices, times)    
    # Run simulation
    network.run(duration, recompile=True)
    # Readout activity
    tot_num_samples = num_samples*len(modulations)
    X, bins, edges = readout(network, connectivity, ro_time, N, tot_num_samples, bin_size=5)
    # Measure reservoir perfomance
    accuracy = classify(X, Y)
    s = np.max(accuracy)
    # Plot
    if plot:
        if exp_dir is None:
            exp_dir = directory
            plots_dir = '{}/plots'.format(exp_dir)
        else:
            plots_dir = '{}/plots/{}'.format(exp_dir, title)
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        if plot['raster']:
            plot_raster(network['mRes'], plots_dir)
        if plot['result']:
            plot_result(X, Y, bins, edges, modulations, snr, directory=plots_dir)
        if plot['network']:
            labels = ['I.up', 'I.dn', 'Q.up', 'Q.dn', 'stop']
            plot_network(connectivity, labels, directory=plots_dir)
        if plot['weights']:
            plot_weights(network, connectivity, N, directory=plots_dir)
        if plot['weights3D']:
            plot_weights_3D(network, connectivity, N, Ngx, Ngy, Ngz, directory=plots_dir)
        if plot['similarity']:
            max_bin = np.argmax(accuracy)
            S = cosine_similarity(list(map(lambda x: x[max_bin], X)))
            labels = [mod for _, mod in sorted(zip(np.unique(Y), modulations), key=lambda pair: pair[0])]
            plot_similarity(S, Y, labels, directory=plots_dir)
        if plot['currents']:
            plot_currents_distributions(network, directory=plots_dir)
        if plot['accuracy']:
            plot_accuracy(accuracy, directory=plots_dir)
    if store:
        result_dir = '{}/results'.format(exp_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        store_result(X, Y, s, params, title=title, directory=result_dir)
    # Remove device folder
    if remove_device:
        shutil.rmtree(directory, ignore_errors=True)
    print("- experiment took {} [s]".format(time.perf_counter()-start))
    return s

