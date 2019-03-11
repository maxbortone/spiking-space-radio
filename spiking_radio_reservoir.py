import time, joblib
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grs
import matplotlib.colors as clr
import matplotlib.cm as cmx
import matplotlib.ticker as ticker
from mpl_toolkits import mplot3d
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
from utils.dataset import *
from utils.reservoir import getTauCurrent
from brian2 import pA, ms, us, SpikeMonitor, StateMonitor, SpikeGeneratorGroup, prefs, device, set_device, defaultclock
from teili import TeiliNetwork
from teili.core.groups import Neurons, Connections
from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


def _pAB(a, b, AoC, DoC):
    if a[3]==1 and b[3]==1:
        C = AoC[0]
    elif a[3]==1 and b[3]==-1:
        C = AoC[1]
    elif a[3]==-1 and b[3]==1:
        C = AoC[2]
    elif a[3]==-1 and b[3]==-1:
        C = AoC[3]
    return C*np.exp(-1*(np.linalg.norm(a[:3]-b[:3])/DoC)**2)

def _wAB(a, b, loc_wResE, scale_wResE, loc_wResI, scale_wResI):
    if a[3]==-1 and b[3]==1:
        w = 1
        while w>0:
            w = -1*np.random.normal(loc_wResI, scale_wResI)
    else:
        w = -1
        while w<0:
            w = np.random.normal(loc_wResE, scale_wResE)
    return w

def setup_connectivity(N, pInh, pIR, Ngx, Ngy, Ngz, AoC, DoC, loc_wResE, scale_wResE, loc_wResI, scale_wResI, rebalance=False):
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

    Ngz : int
        number of reservoir neurons in the z-axis of the grid

    AoC : list
        list of probability amplitudes for the connections between
        reservoir neurons ([ex-ex, ex-inh, inh-ex, inh-inh])
    
    DoC : float
        density of connections between reservoir neurons

    loc_wResE : float
        mean value of the reservoir excitatory weights distribution

    scale_wResE : float
        standard deviation of the reservoir excitatory weights distribution

    loc_wResI : float
        mean value of the reservoir inhibitory weights distribution

    scale_wResI : float
        standard deviation of the reservoir inhibitory weights distribution

    rebalance : bool
        wheter the reservoir weights should be adjusted to satisfy the
        input balance condition for each neuron

    Returns
    -------
    connectivity : object
        contains the two connectivity matrices as
        i and j indices to be used in the connect method
        of the synapse object in Brian2
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
        a = np.array(list(zip(*np.where(grid[:,:,:,0]==n)))[0])
        a = np.append(a, grid[a[0], a[1], a[2], 1])
        for x in range(Ngx):
            for y in range(Ngy):
                for z in range(Ngz):
                    b = np.array([x, y, z, grid[x, y, z, 1]])
                    p = _pAB(a, b, AoC, DoC)
                    r = np.random.uniform()
                    if r<p:
                        Cres[0].append(n)
                        Cres[1].append(grid[x, y, z, 0])
                        Wres.append(_wAB(a, b, loc_wResE, scale_wResE, loc_wResI, scale_wResI))
    # rebalance weights if needed
    if rebalance:
        Wres_new = np.zeros((N, N))
        Wres_new[Cres[0], Cres[1]] = Wres
        for i in neurons:
            I = np.where(Wres_new[:, i]<0)[0]
            if len(I)>0:
                delta = -np.sum(Wres_new[:,i])/len(I)
                Wres_new[I, i] += delta
            E = np.where(Wres_new[:, i]>0)[0]
            if len(E)>0:
                delta = -np.sum(Wres_new[:,i])/len(E)
                Wres_new[E, i] += delta
        Wres = Wres_new.flatten() 
    # connect input to reservoir
    Cin = [[], []]
    for n in neurons:
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
            'i': np.array(Cin[0]),
            'j': np.array(Cin[1])
        },
        'res_res': {
            'i': np.array(Cres[0]),
            'j': np.array(Cres[1]),
            'w': np.array(Wres)
        },
        'grid': grid
    }
    return connectivity

def setup_input_layer(components, wGen):
    """
    Setup the input layer consisting of a spike generator with 4 neurons
    that project to 2 input neurons through excitatory and inhibitory synapsis

    Parameters
    ----------
    components : object
        previously defined network components

    wGen : float
        weight of the generator synapsis

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
    sGenInp.weight = wGen*np.array([1.0, -1.0, 1.0, -1.0])
    mGen = SpikeMonitor(gGen, name='mGen')
    mInp = SpikeMonitor(gInp, name='mInp')
    components['generator'] = gGen
    components['layers'] = {'gInp': gInp}
    components['synapsis'] = {'sGenInp': sGenInp}
    components['monitors'] = {'mGen': mGen, 'mInp': mInp}
    return components

def setup_reservoir_layer(components, connectivity, N, Itau, wInp):
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

    N : int
        number of neurons in the reservoir

    Itau : float
        current of the membrane potential decay time of
        the reservoir neurons in pA

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
    sResRes.weight = connectivity['res_res']['w']
    #sResRes.I_etau = Ietau
    #sResRes.I_itau = Iitau 
    sInpRes = Connections(components['layers']['gInp'], gRes, equation_builder=DPISyn(), method='euler', name='sInpRes')
    sInpRes.connect(i=connectivity['inp_res']['i'], j=connectivity['inp_res']['j'])
    sInpRes.weight = wInp
    mRes = SpikeMonitor(gRes, name='mRes')
    recorded_synapsis = np.random.randint(0, high=len(connectivity['res_res']['i']), size=10)
    smRes = StateMonitor(sResRes, ['Ie_syn', 'Ii_syn'], name='smRes', record=recorded_synapsis)
    components['layers']['gRes'] = gRes
    components['synapsis']['sInpRes'] = sInpRes
    components['synapsis']['sResRes'] = sResRes
    components['monitors']['mRes'] = mRes
    components['monitors']['smRes'] = smRes
    return components

def init_network(indices, times, connectivity, N, Itau, wGen, wInp):
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
        i and j indices to be used in the connect method
        of the synapse object in Brian2

    N : int
        number of neurons in the reservoir

    tau : float
        membrane potential decay time in ms of the reservoir
        neurons

    wGen : float
        weight of the generator synapsis

    wInp : float
        weight of the input synapsis
    
    loc_wRes : float
        mean value of the reservoir weights distribution

    scale_wRes : float
        standard deviation of the reservoir weights distribution

    Returns
    -------
    network : TeiliNetwork
        instance of the network with all the commponents
    """
    # setup input layer
    components = {'generator': None, 'layers': None, 'synapsis': None, 'monitors': None}
    components = setup_input_layer(components, wGen)
    # setup reservoir layer
    components = setup_reservoir_layer(components, connectivity, N, Itau, wInp)
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
    X = list(map(lambda x: x.T.flatten(), X))
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
    try:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        Y_pred = kmeans.labels_
        score = silhouette_score(X, Y_pred)
    except:
        score = -1.0
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

def plot_result(X, Y, bins, edges, modulations, snr, directory=None):
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

    directory : string
        path to the folder into which the plot should be saved
    """
    num_samples = len(X)
    num_classes = len(np.unique(Y))
    num_samples_per_class = int(num_samples/num_classes)
    for i in range(num_samples):
        sid = i%num_samples_per_class
        fig= plt.figure()
        x = X[i].reshape((bins[1], bins[0]))
        plt.imshow(x, interpolation='nearest', origin='low', aspect='auto', \
           extent=[0, bins[0], 0, bins[1]], cmap='viridis')
        plt.title('{} @ {} #{}'.format(modulations[Y[i]], snr, sid))
        plt.xlabel('vector element')
        plt.ylabel('neuron index')
        if directory:
            plt.savefig(directory+'/{}_{}_{}.pdf'.format(modulations[Y[i]], snr, sid), bbox_inches='tight')
            plt.close(fig=fig)


def plot_network(network, N, weights, directory=None):
    """
    Plot the network layers and connections

    Parameters
    ----------
    network : TeiliNetwork
        instance of the network with all the components

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
    ax1.tick_params(direction='in')
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(['I.up', 'I.dn', 'Q.up', 'Q.dn'])
    ax1.set_yticks([0, 1])
    ax1.set_title('Generator')
    ax2.scatter(network['sInpRes'].i, network['sInpRes'].j, c='k', marker='.')
    ax2.set_xlabel('Source neuron index')
    ax2.set_xticks([0, 1])
    ax2.tick_params(direction='in')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax2.set_yticklabels([])
    ax2.set_title('Input')
    C = np.zeros((N, N))
    C[network['sResRes'].i, network['sResRes'].j] = weights
    ax3.imshow(C, interpolation='nearest', origin='low', aspect='auto', \
                extent=[0, N, 0, N], cmap='viridis', vmin=weights.min(), vmax=weights.max())
    ax3.set_xlabel('Source neuron index')
    ax3.tick_params(direction='in')
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax3.set_title('Reservoir')
    if directory:
        plt.savefig(directory+'/network.pdf', bbox_inches='tight')
        plt.close(fig=fig)

def plot_weights(network, connectivity, N, Ngx, Ngy, directory=None):
    """
    Plot the network weights for each neuron

    Parameters
    ----------
    network : TeiliNetwork
        instance of the network with all the commponents

    connectivity : object
        contains the two connectivity matrices as
        i and j indices to be used in the connect method
        of the synapse object in Brian2

    N : int
        number of neurons in the reservoir

    Ngx : int
        number of reservoir neurons in the x-axis of the grid

    Ngy : int
        number of reservoir neurons in the y-axis of the grid

    directory : string
        path to the folder into which the plot should be saved
    """
    source = connectivity['res_res']['i']
    target = connectivity['res_res']['j']
    weight = connectivity['res_res']['w']
    types = connectivity['types']
    grid = connectivity['grid']
    fig = plt.figure()
    grd = grs.GridSpec(Ngx, Ngy, wspace=0.0, hspace=0.0)
    for n in range(N):
        W = np.zeros((Ngx, Ngy))
        t = target[np.where(source==n)[0]]
        coords = np.array(list(map(lambda m: tuple(zip(*np.where(grid==m)))[0], t)))
        W[coords[:, 0], coords[:, 1]] = weight[np.where(source==n)[0]]
        ax = plt.Subplot(fig, grd[n])
        im = ax.imshow(W.T, interpolation='nearest', origin='low', aspect='auto', \
                extent=[0, Ngx, 0, Ngy], cmap='viridis', vmin=weight.min(), vmax=weight.max())
        ax.set_yticks([])
        ax.set_xticks([])
        sx, sy = tuple(zip(*np.where(grid==n)))[0]
        if types[n]==1:
            ax.annotate("{}".format(n), xy=[sx, sy], fontsize=2, color='red')
        else:
            ax.annotate("{}".format(n), xy=[sx, sy], fontsize=2, color='white')
        fig.add_subplot(ax)
    ax_cbar1 = fig.add_axes([1, 0.1, 0.05, 0.8])
    plt.colorbar(im, cax=ax_cbar1, orientation='vertical', label='weight')
    if directory:
        plt.savefig(directory+'/weights.pdf', bbox_inches='tight')
        plt.close(fig=fig)

def plot_weights_3D(network, connectivity, N, Ngx, Ngy, Ngz, directory=None):
    """
    Plot the network weights for each neuron

    Parameters
    ----------
    network : TeiliNetwork
        instance of the network with all the commponents

    connectivity : object
        contains the two connectivity matrices as
        i and j indices to be used in the connect method
        of the synapse object in Brian2

    N : int
        number of neurons in the reservoir

    Ngx : int
        number of reservoir neurons in the x-axis of the grid

    Ngy : int
        number of reservoir neurons in the y-axis of the grid

    Ngz : int
        number of reservoir neurons in the z-axis of the grid

    directory : string
        path to the folder into which the plot should be saved
    """
    source = connectivity['res_res']['i']
    target = connectivity['res_res']['j']
    weight = connectivity['res_res']['w']
    grid = connectivity['grid']
    fig = plt.figure()
    norm1  = clr.Normalize(vmin=weight.min(), vmax=weight.max())
    smap1 = cmx.ScalarMappable(norm=norm1, cmap='viridis')
    norm2  = clr.Normalize(vmin=0, vmax=9)
    smap2 = cmx.ScalarMappable(norm=norm2, cmap='tab10')
    mx, my, mz = np.meshgrid(np.arange(Ngx), np.arange(Ngy), np.arange(Ngz))
    os.makedirs(directory+'/weights')
    for n in range(N):
        ax = plt.axes(projection='3d')
        sx, sy, sz = tuple(zip(*np.where(grid[:,:,:,0]==n)))[0]
        t = target[np.where(source==n)[0]]
        c = np.array(list(map(lambda m: tuple(zip(*np.where(grid[:,:,:,0]==m)))[0], t)))
        w = weight[np.where(source==n)[0]]
        colors = np.zeros((Ngx, Ngy, Ngz))
        exc = np.array(list(zip(*np.where(grid[:,:,:,1]==1))))
        inh = np.array(list(zip(*np.where(grid[:,:,:,1]==-1))))
        colors[exc[:,0], exc[:,1], exc[:,2]] = 3
        colors[inh[:,0], inh[:,1], inh[:,2]] = 0
        colors[sx, sy, sz] = 2
        for i in range(len(t)):
            ax.plot3D([sy, c[i][1]], [sx, c[i][0]], [sz, c[i][2]], color=smap1.to_rgba(w[i]))
        ax.text(sy, sx, sz, "{}-{}".format(grid[sx, sy, sz, 0], 'e' if grid[sx, sy, sz, 1]==1 else 'i' ), color='k')
        ax.scatter3D(mx, my, mz, c=smap2.to_rgba(colors.flatten()))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        if directory:
            plt.savefig(directory+'/weights/weights_n{}.pdf'.format(n), bbox_inches='tight')
            plt.close(fig=fig)

def plot_similarity(X, Y, modulations, directory=None):
    """
    Plot the cosine similarity matrix between the reservoir 
    readouts of the samples

    Parameters
    ----------
    X : ndarray (num_samples, num_features)
        readout output from the reservoir for each sample
        in the stimulus

    Y : ndarray (num_samples)
        labels for each sample in the stimulus

    modulations : list
        modulation classes in the input stimulus

    directory : string
        path to the folder into which the plot should be saved
    """
    X = [x for _, x in sorted(zip(Y, X), key=lambda pair: pair[0])]
    S = cosine_similarity(X)
    num_samples = len(X)
    num_classes = len(np.unique(Y))
    num_samples_per_class = int(num_samples/num_classes)
    ticks = [(i+1)*num_samples_per_class for i in range(len(modulations))]
    labels = [mod for _, mod in sorted(zip(np.unique(Y), modulations), key=lambda pair: pair[0])]
    fig, ax = plt.subplots()
    im = ax.imshow(S.T, interpolation='nearest', origin='low', aspect='auto', \
           extent=[0, num_samples, 0, num_samples], cmap='viridis')
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax_cbar1 = fig.add_axes([1, 0.1, 0.05, 0.8])
    plt.colorbar(im, cax=ax_cbar1, orientation='vertical', label='similarity')
    if directory:
        plt.savefig(directory+'/similarity.pdf', bbox_inches='tight')
        plt.close(fig=fig)

def plot_currents(monitor, connectivity, directory=None):
    """
    Plot the some randomly chosen synaptic currents

    Parameters
    ----------
    monitor : StateMonitor
        state monitor of the reservoir

    connectivity : object
        contains the two connectivity matrices as
        i and j indices to be used in the connect method
        of the synapse object in Brian2

    directory : string
        path to the folder into which the plot should be saved
    """
    recorded_synapsis = monitor.record
    print("Synapse: ", recorded_synapsis)
    fig = plt.figure()
    grd = grs.GridSpec(len(recorded_synapsis), 1, wspace=0.0, hspace=0.0)
    for i in range(len(recorded_synapsis)):
        ax = plt.Subplot(fig, grd[i])
        ax.plot(monitor.t/ms, monitor.Ie_syn[i], color='red', linestyle='solid', label="Ie_syn")
        ax.plot(monitor.t/ms, monitor.Ii_syn[i], color='blue', linestyle='dashed', label="Ii_syn")
        fig.add_subplot(ax)
    if directory:
        plt.savefig(directory+'/currents.pdf', bbox_inches='tight')
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
def experiment(wGen=3500, wInp=3500, loc_wResE=50, scale_wResE=10, loc_wResI=50, scale_wResI=10, 
    pIR=0.3, pInh=0.2, AoC=[0.3, 0.2, 0.5, 0.1], DoC=2, \
    N=200, tau=20, Ngx=5, Ngy=5, Ngz=8, \
    indices=None, times=None, stretch_factor=None, duration=None, ro_time=None, \
    modulations=None, snr=None, num_samples=None, Y=None, \
    plot=False, store=False, title=None, exp_dir=None, remove_device=False):
    """
    Run an experiment on a reservoir with given properties and evaluate
    its performance in terms of clustering inputs of different classes.

    Parameters
    ----------
    wGen : float
        weight of the generator synapsis

    wInp : float
        weight of the input synapsis
    
    loc_wResE : float
        mean value of the reservoir excitatory weights distribution

    scale_wResE : float
        standard deviation of the reservoir excitatory weights distribution

    loc_wResI : float
        mean value of the reservoir inhibitory weights distribution

    scale_wResI : float
        standard deviation of the reservoir inhibitory weights distribution

    pIR : float
        probability of connection for the input neurons

    pInh : float
        probability that a reservoir neuron is inhibitory

    AoC : list
        list of probability amplitudes for the connections between
        reservoir neurons ([ex-ex, ex-inh, inh-ex, inh-inh])

    N : int
        number of neurons in the reservoir

    tau : float
        membrane potential decay time in ms of the reservoir
        neurons

    Ngx : int
        number of reservoir neurons in the x-axis of the grid

    Ngy : int
        number of reservoir neurons in the y-axis of the grid

    Ngz : int
        number of reservoir neurons in the z-axis of the grid

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
    print("- running with: wGen={}, wInp={}, loc_wResE={}, scale_wResE={}, loc_wResI={}, scale_wResI={}".format(wGen, wInp, \
        loc_wResE, scale_wResE, loc_wResI, scale_wResI))
    # Setup connectivity of the network
    # and the neurons time constant
    connectivity = setup_connectivity(N, pInh, pIR, Ngx, Ngy, Ngz, AoC, DoC, loc_wResE, scale_wResE, loc_wResI, scale_wResI)
    Itau = getTauCurrent(tau*ms)
    # Set C++ backend and time step
    if title==None:
        title = 'srres_{}'.format(os.getpid())
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
    tot_num_samples = num_samples*len(modulations)
    X, bins, edges = readout(network['mRes'], ro_time, N, tot_num_samples, bin_size=5)
    # Plot
    if plot:
        if exp_dir==None:
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
            plot_network(network, N, connectivity['res_res']['w'], directory=plots_dir)
        if plot['weights']:
            plot_weights_3D(network, connectivity, N, Ngx, Ngy, Ngz, directory=plots_dir)
        if plot['similarity']:
            plot_similarity(X, Y, modulations, directory=plots_dir)
        if plot['currents']:
            plot_currents(network['smRes'], connectivity, directory=plots_dir)
    # Measure reservoir perfomance
    s = classify(X, Y)
    if store:
        store_result(X, Y, s, params)
    # Remove device folder
    if remove_device:
        shutil.rmtree(directory, ignore_errors=True)
    print("- experiment took {} [s]".format(time.perf_counter()-start))
    return s

if __name__ == '__main__':
    from utils.modulator import AsynchronousDeltaModulator

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
    dataset = load_dataset('./data/radioML/RML2016.10a_dict.pkl', snr=snr, normalize=True)
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
    plot_flags = {
        'raster': False,
        'result': True,
        'network': True,
        'weights': True,
        'similarity': True,
        'currents': False
    }
    score = experiment(wGen=3500, wInp=3500, loc_wResE=1050, scale_wResE=105, loc_wResI=1050, scale_wResI=105, 
        pIR=0.1, pInh=0.2, AoC=[0.3, 0.2, 0.5, 0.1], DoC=2, \
        N=200, tau=20, Ngx=5, Ngy=5, Ngz=8, \
        indices=indices, times=times, stretch_factor=stretch_factor, duration=duration, ro_time=stimulation+pause, \
        modulations=modulations, snr=snr, num_samples=num_samples, Y=Y, \
        plot=plot_flags, store=False)
    print(score)
