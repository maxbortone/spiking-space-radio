import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grs
import matplotlib.colors as clr
import matplotlib.cm as cmx
import matplotlib.ticker as ticker
from mpl_toolkits import mplot3d
from matplotlib.lines import Line2D
from brian2 import ms, nA, pA

def plot_stimulus(indices, times, ro_time, num_samples, figsize=(6, 4), directory=None):
    """
    Plot input stimulus

    Parameters
    ----------
    indices : list
        spike generator neuron indices

    times : list
        spike generator spiking times

    ro_time : float
        width of the time interval from which to read out
        the activity of one sample in ms

    num_samples : int
        number of samples

    
    """
    fig = plt.figure(figsize=figsize)
    grid = grs.GridSpec(2, 1, wspace=0.0, hspace=0.0)
    ax1 = plt.Subplot(fig, grid[0])
    ax1.set_yticks([-0.5, 0.5])
    ax1.set_yticklabels(['I.dn', 'I.up'])
    ax1.vlines(times[indices==0]/ms, ymin=0.0, ymax=1.0, colors='#e74c3c', linewidth=1)
    ax1.vlines(times[indices==1]/ms, ymin=-1.0, ymax=0.0, colors='#3498db', linewidth=1)
    ax2 = plt.Subplot(fig, grid[1], sharex=ax1)
    ax2.set_yticks([-0.5, 0.5])
    ax2.set_yticklabels(['Q.dn', 'Q.up'])
    ax2.set_xlabel(r'time [ms]', fontsize=18)
    ax2.vlines(times[indices==2]/ms, ymin=0.0, ymax=1.0, colors='#e74c3c', linewidth=1)
    ax2.vlines(times[indices==3]/ms, ymin=-1.0, ymax=0.0, colors='#3498db', linewidth=1)
    bins = [ro_time*(i+1) for i in range(num_samples)]
    ax1.vlines(bins/ms, ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], colors="#95a5a6", linewidth=1, zorder=0)
    ax2.vlines(bins/ms, ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1], colors="#95a5a6", linewidth=1, zorder=0)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    if directory:
        plt.savefig(directory+'/stimulus.pdf', bbox_inches='tight')
        plt.close(fig=fig)

def plot_sample(signal, time_stim, indices, times, title=None, figsize=(6, 4), directory=None, name='sample'):
    """
    Plot 

    Parameters
    ----------
    """
    fig = plt.figure(figsize=figsize)
    grid = grs.GridSpec(3, 1, wspace=0.0, hspace=0.0)
    ax1 = plt.Subplot(fig, grid[0])
    ax1.set_title(title)
    ax1.plot(time_stim/ms, signal[0, :], color='#2ecc71', label='I')
    ax1.plot(time_stim/ms, signal[1, :], color='#e67e22', label='Q')
    ax1.legend(loc='upper right')
    ax2 = plt.Subplot(fig, grid[1], sharex=ax1)
    ax2.set_yticks([-0.5, 0.5])
    ax2.set_yticklabels(['I.dn', 'I.up'])
    ax2.vlines(times[indices==0]/ms, ymin=0.0, ymax=1.0, colors='#e74c3c', linewidth=1)
    ax2.vlines(times[indices==1]/ms, ymin=-1.0, ymax=0.0, colors='#3498db', linewidth=1)
    ax3 = plt.Subplot(fig, grid[2], sharex=ax1)
    ax3.set_yticks([-0.5, 0.5])
    ax3.set_yticklabels(['Q.dn', 'Q.up'])
    ax3.set_xlabel(r'time [ms]', fontsize=18)
    ax3.vlines(times[indices==2]/ms, ymin=0.0, ymax=1.0, colors='#e74c3c', linewidth=1)
    ax3.vlines(times[indices==3]/ms, ymin=-1.0, ymax=0.0, colors='#3498db', linewidth=1)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    fig.add_subplot(ax3)
    if directory:
        plt.savefig(directory+'/{}.pdf'.format(name), bbox_inches='tight')
        plt.close(fig=fig)

def plot_sample_with_labels(signal, time_stim, indices, times, labels, time_labels, bins, \
    title=None, figsize=(6, 4), directory=None, name='sample_with_labels'):
    """
    Plot 

    Parameters
    ----------
    """
    fig = plt.figure(figsize=figsize)
    grid = grs.GridSpec(4, 1, wspace=0.0, hspace=0.0)
    ax1 = plt.Subplot(fig, grid[0])
    ax1.set_title(title)
    ax1.plot(time_stim/ms, signal[0, :], color='#2ecc71', label='I')
    ax1.plot(time_stim/ms, signal[1, :], color='#e67e22', label='Q')
    ax1.legend(loc='upper right')
    ax1.vlines(bins/ms, ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], colors="#95a5a6", linewidth=1)
    ax2 = plt.Subplot(fig, grid[1], sharex=ax1)
    ax2.set_yticks([-0.5, 0.5])
    ax2.set_yticklabels(['I.dn', 'I.up'])
    ax2.vlines(times[indices==0]/ms, ymin=0.0, ymax=1.0, colors='#e74c3c', linewidth=1)
    ax2.vlines(times[indices==1]/ms, ymin=-1.0, ymax=0.0, colors='#3498db', linewidth=1)
    ax2.vlines(bins/ms, ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1], colors="#95a5a6", linewidth=1)
    ax3 = plt.Subplot(fig, grid[2], sharex=ax1)
    ax3.set_yticks([-0.5, 0.5])
    ax3.set_yticklabels(['Q.dn', 'Q.up'])
    ax3.vlines(times[indices==2]/ms, ymin=0.0, ymax=1.0, colors='#e74c3c', linewidth=1)
    ax3.vlines(times[indices==3]/ms, ymin=-1.0, ymax=0.0, colors='#3498db', linewidth=1)
    ax3.vlines(bins/ms, ymin=ax3.get_ylim()[0], ymax=ax3.get_ylim()[1], colors="#95a5a6", linewidth=1)
    ax4 = plt.Subplot(fig, grid[3], sharex=ax1)
    ax4.plot(time_labels/ms, labels, color="#3498db")
    ax4.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax4.set_ylabel("label", fontsize=18)
    ax4.set_xlabel(r'time [ms]', fontsize=18)
    ax4.vlines(bins/ms, ymin=ax4.get_ylim()[0], ymax=ax4.get_ylim()[1], colors="#95a5a6", linewidth=1)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    fig.add_subplot(ax3)
    fig.add_subplot(ax4)
    if directory:
        plt.savefig(directory+'/{}.pdf'.format(name), bbox_inches='tight')
        plt.close(fig=fig)
    
def plot_sample_with_reconstruction(signal, reconstruction, time_stim, indices, times, \
    title=None, figsize=(6, 4), directory=None, name='sample_with_reconstruction'):
    """
    Plot 

    Parameters
    ----------
    """
    fig = plt.figure(figsize=figsize)
    grid = grs.GridSpec(3, 1, wspace=0.0, hspace=0.0)
    ax1 = plt.Subplot(fig, grid[0])
    ax1.set_title(title)
    ax1.plot(time_stim/ms, signal[0, :], color='#2ecc71', label='I')
    ax1.plot(time_stim/ms, signal[1, :], color='#e67e22', label='Q')
    ax1.plot(time_stim/ms, reconstruction[0, :], color='#34495e', linestyle='dashed', alpha=0.6, label="reconstruction")
    ax1.plot(time_stim/ms, reconstruction[1, :], color='#34495e', linestyle='dashed', alpha=0.6)
    ax1.legend(loc='upper right')
    ax2 = plt.Subplot(fig, grid[1], sharex=ax1)
    ax2.set_yticks([-0.5, 0.5])
    ax2.set_yticklabels(['I.dn', 'I.up'])
    ax2.vlines(times[indices==0]/ms, ymin=0.0, ymax=1.0, colors='#e74c3c', linewidth=1)
    ax2.vlines(times[indices==1]/ms, ymin=-1.0, ymax=0.0, colors='#3498db', linewidth=1)
    ax3 = plt.Subplot(fig, grid[2], sharex=ax1)
    ax3.set_yticks([-0.5, 0.5])
    ax3.set_yticklabels(['Q.dn', 'Q.up'])
    ax3.set_xlabel(r'time [ms]', fontsize=18)
    ax3.vlines(times[indices==2]/ms, ymin=0.0, ymax=1.0, colors='#e74c3c', linewidth=1)
    ax3.vlines(times[indices==3]/ms, ymin=-1.0, ymax=0.0, colors='#3498db', linewidth=1)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    fig.add_subplot(ax3)
    if directory:
        plt.savefig(directory+'/{}.pdf'.format(name), bbox_inches='tight')
        plt.close(fig=fig)
    
def plot_input_synapsis(stateMon, spikeMon, weight, title, directory=None):
    """
    Plot 

    Parameters
    ----------
    """
    fig = plt.figure()
    grid = grs.GridSpec(4, 1, wspace=0.0, hspace=0.0)
    ax1 = plt.Subplot(fig, grid[0])
    ax1.set_title(title)
    if weight[0]>0:
        ax1.plot(stateMon.t/ms, stateMon.Ie_syn[0], color='#e74c3c')
    else:
        ax1.plot(stateMon.t/ms, stateMon.Ii_syn[0], color='#e74c3c')
    ax1.set_ylabel('I.up')
    spikes = spikeMon.i==0
    ax1.scatter(spikeMon.t[spikes]/ms, np.zeros(len(spikeMon.i[spikes])), c='k', marker='.')
    ax2 = plt.Subplot(fig, grid[1])
    if weight[1]>0:
        ax2.plot(stateMon.t/ms, stateMon.Ie_syn[1], color='#3498db')
    else:
        ax2.plot(stateMon.t/ms, stateMon.Ii_syn[1], color='#3498db')
    ax2.set_ylabel('I.dn')
    spikes = spikeMon.i==1
    ax2.scatter(spikeMon.t[spikes]/ms, np.zeros(len(spikeMon.i[spikes])), c='k', marker='.')
    ax3 = plt.Subplot(fig, grid[2])
    if weight[2]>0:
        ax3.plot(stateMon.t/ms, stateMon.Ie_syn[2], color='#e74c3c')
    else:
        ax3.plot(stateMon.t/ms, stateMon.Ii_syn[2], color='#e74c3c')
    ax3.set_ylabel('Q.up')
    spikes = spikeMon.i==2
    ax3.scatter(spikeMon.t[spikes]/ms, np.zeros(len(spikeMon.i[spikes])), c='k', marker='.')
    ax4 = plt.Subplot(fig, grid[3])
    if weight[3]>0:
        ax4.plot(stateMon.t/ms, stateMon.Ie_syn[3], color='#3498db')
    else:
        ax4.plot(stateMon.t/ms, stateMon.Ii_syn[3], color='#3498db')
    ax4.set_ylabel('Q.dn')
    spikes = spikeMon.i==3
    ax4.scatter(spikeMon.t[spikes]/ms, np.zeros(len(spikeMon.i[spikes])), c='k', marker='.')
    ax4.set_xlabel('time [ms]', fontsize=18)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    fig.add_subplot(ax3)
    fig.add_subplot(ax4)
    if directory:
        plt.savefig(directory+'/input_synapsis.pdf', bbox_inches='tight')
        plt.close(fig=fig)
    
def plot_input_layer(stateMon, spikeMon, title, directory=None):
    """
    Plot 

    Parameters
    ----------
    """
    fig = plt.figure()
    grid = grs.GridSpec(2, 1, wspace=0.0, hspace=0.0)
    ax1 = plt.Subplot(fig, grid[0])
    ax1.set_title(title)
    ax1.plot(stateMon.t/ms, stateMon.Iin[0], color='#2ecc71')
    ax1.set_ylabel('I')
    spikes = spikeMon.i==0
    ax1.scatter(spikeMon.t[spikes]/ms, np.zeros(len(spikeMon.i[spikes])), c='k', marker='.')
    ax2 = plt.Subplot(fig, grid[1])
    ax2.plot(stateMon.t/ms, stateMon.Iin[1], color='#e67e22')
    ax2.set_ylabel('Q')
    spikes = spikeMon.i==1
    ax2.scatter(spikeMon.t[spikes]/ms, np.zeros(len(spikeMon.i[spikes])), c='k', marker='.')
    ax2.set_xlabel('time [ms]', fontsize=18)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    if directory:
        plt.savefig(directory+'/input_layer.pdf', bbox_inches='tight')
        plt.close(fig=fig)
    
def plot_input_layer_with_membrane_potential(indices, times, network, duration, \
    bins=None, figsize=(6, 4), directory=None):
    """
    Plot 

    Parameters
    ----------
    """
    fig = plt.figure(figsize=figsize)
    grid = grs.GridSpec(3, 1, wspace=0.0, hspace=0.0)
    ax1 = plt.Subplot(fig, grid[0])
    ax1.set_yticks([-0.5, 0.5, 1.5, 2.5, 3.5])
    ax1.set_yticklabels(['stop', 'Q.dn', 'Q.up', 'I.dn', 'I.up'])
    ax1.set_ylabel("generator neurons")
    ax1.vlines(times[indices==0]/ms, ymin=3.0, ymax=4.0, colors='#2ecc71', linewidth=1)
    ax1.vlines(times[indices==1]/ms, ymin=2.0, ymax=3.0, colors='#e74c3c', linewidth=1)
    ax1.vlines(times[indices==2]/ms, ymin=1.0, ymax=2.0, colors='#e67e22', linewidth=1)
    ax1.vlines(times[indices==3]/ms, ymin=0.0, ymax=1.0, colors='#3498db', linewidth=1)
    ax1.vlines(times[indices==4]/ms, ymin=-1.0, ymax=0.0, colors='#8e44ad', linewidth=1)
    ax2 = plt.Subplot(fig, grid[1], sharex=ax1)
    ax2.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax2.set_yticklabels([3, 2, 1, 0])
    ax2.set_ylabel("input neurons")
    ax2.vlines(network['mInp'].t[network['mInp'].i==0]/ms, ymin=3.0, ymax=4.0, colors='#2ecc71', linewidth=1)
    ax2.vlines(network['mInp'].t[network['mInp'].i==1]/ms, ymin=2.0, ymax=3.0, colors='#e74c3c', linewidth=1)
    ax2.vlines(network['mInp'].t[network['mInp'].i==2]/ms, ymin=1.0, ymax=2.0, colors='#e67e22', linewidth=1)
    ax2.vlines(network['mInp'].t[network['mInp'].i==3]/ms, ymin=0.0, ymax=1.0, colors='#3498db', linewidth=1)
    ax3 = plt.Subplot(fig, grid[2], sharex=ax1)
    ax3.plot(network['smInp'].t/ms, network['smInp'].Imem[0]/nA, c='#2ecc71', label="0")
    ax3.plot(network['smInp'].t/ms, network['smInp'].Imem[1]/nA, c='#e74c3c', label="1")
    ax3.plot(network['smInp'].t/ms, network['smInp'].Imem[2]/nA, c='#e67e22', label="2")
    ax3.plot(network['smInp'].t/ms, network['smInp'].Imem[3]/nA, c='#3498db', label="3")
    ax3.axhline(network['gInp'].Ispkthr[0]/nA, linestyle='dashed', linewidth=1, color='#95a5a6', label="Ispkthr")
    ax3.set_ylabel("membrane potential")
    ax3.set_xlabel("time [ms]")
    ax3.legend(loc='best')
    ax3.set_xlim((0, np.ceil(duration/ms)))
    if bins is not None:
        ax1.vlines(bins/ms, ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], colors="#95a5a6", linewidth=1, zorder=0)
        ax2.vlines(bins/ms, ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1], colors="#95a5a6", linewidth=1, zorder=0)
        ax3.vlines(bins/ms, ymin=ax3.get_ylim()[0], ymax=ax3.get_ylim()[1], colors="#95a5a6", linewidth=1, zorder=0)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    fig.add_subplot(ax3)
    if directory:
        plt.savefig(directory+'/input_layer_with_membrane_potential.pdf', bbox_inches='tight')
        plt.close(fig=fig)

def plot_raster(monitor, figsize=(6, 4), directory=None):
    """
    Plot the raster of spikes generated by the input stimulus

    Parameters
    ----------
    monitor : SpikeMonitor
        spike monitor of the reservoir
    """
    # TODO: add pagination for long stimuli
    fig = plt.figure(figsize=figsize)
    plt.scatter(monitor.t/ms, monitor.i, marker=',', s=2)
    plt.xlabel('time [ms]')
    plt.ylabel('reservoir neuron')
    if directory:
        plt.savefig(directory+'/raster_plot.pdf', bbox_inches='tight')
        plt.close(fig=fig)
    
def plot_reservoir_raster(connectivity, params, network, times, duration, \
    bins=None, figsize=(6, 4), directory=None):
    """
    Plot 

    Parameters
    ----------
    """
    input_neurons_Iup = connectivity['inp_res']['j'][connectivity['inp_res']['i']==0]
    input_neurons_Idn = connectivity['inp_res']['j'][connectivity['inp_res']['i']==1]
    input_neurons_Qup = connectivity['inp_res']['j'][connectivity['inp_res']['i']==2]
    input_neurons_Qdn = connectivity['inp_res']['j'][connectivity['inp_res']['i']==3]
    colors = []
    for n in range(params['N']):
        if n in input_neurons_Iup:
            colors.append('#2ecc71')
        elif n in input_neurons_Idn:
            colors.append('#e74c3c')
        elif n in input_neurons_Qup:
            colors.append('#e67e22')
        elif n in input_neurons_Qdn:
            colors.append('#3498db')
        elif connectivity['types'][n]==1:
            colors.append('#8e44ad')
        else:
            colors.append('#2c3e50')
    positions = []
    for n in range(params['N']):
        idx = np.where(network['mRes'].i==n)[0]
        positions.append(network['mRes'].t[idx]/ms)
    fig, ax = plt.subplots(figsize=figsize)
    ax.eventplot(positions, colors=colors)
    ax.axvline(times[-1]/ms, linestyle='dashed', color='#95a5a6')
    ax.invert_yaxis()
    ax.set_ylabel("neuron index")
    ax.set_xlabel("time [ms]")
    ax.set_xlim((0, np.ceil(duration/ms)))
    if bins is not None:
        plt.vlines(bins/ms, ymin=0, ymax=params['N'], colors="#95a5a6", linewidth=1, zorder=0)
    lines = [Line2D([0], [0], color='#2ecc71', lw=2),
             Line2D([0], [0], color='#e74c3c', lw=2),
             Line2D([0], [0], color='#e67e22', lw=2),
             Line2D([0], [0], color='#3498db', lw=2),
             Line2D([0], [0], color='#8e44ad', lw=2),
             Line2D([0], [0], color='#2c3e50', lw=2)]
    handles = ['I.up', 'I.dn', 'Q.up', 'Q.dn', 'exc', 'inh']
    ax.legend(lines, handles, loc="upper right")
    if directory:
        plt.savefig(directory+'/hennequin_reservoir_raster.pdf', bbox_inches='tight')
        plt.close(fig=fig)

def plot_hennequin_reservoir_raster_without_input(connectivity, params, network, times, duration, \
    bins=None, figsize=(6, 4), directory=None):
    """
    Plot 

    Parameters
    ----------
    """
    input_neurons_Iup = connectivity['inp_res']['j'][connectivity['inp_res']['i']==0]
    input_neurons_Idn = connectivity['inp_res']['j'][connectivity['inp_res']['i']==1]
    input_neurons_Qup = connectivity['inp_res']['j'][connectivity['inp_res']['i']==2]
    input_neurons_Qdn = connectivity['inp_res']['j'][connectivity['inp_res']['i']==3]
    mask = np.zeros(params['N'])
    mask[input_neurons_Iup] = 1.0
    mask[input_neurons_Idn] = 1.0
    mask[input_neurons_Qup] = 1.0
    mask[input_neurons_Qdn] = 1.0
    reservoir_neurons = np.ma.array(np.arange(params['N']), mask=mask)
    M = params['Ngx']*params['Ngy']
    colors = []
    for n in range(params['N']):
        if n < M:
            colors.append('#8e44ad')
        else:
            colors.append('#2c3e50')
    positions = []    
    for n in reservoir_neurons:
        idx = np.where(network['mRes'].i==n)[0]
        positions.append(network['mRes'].t[idx]/ms)
    fig, ax = plt.subplots(figsize=figsize)
    ax.eventplot(positions, colors=colors)
    ax.axvline(times[-1]/ms, linestyle='dashed', color='#95a5a6')
    ax.invert_yaxis()
    ax.set_ylabel("neuron index")
    ax.set_xlabel("time [ms]")
    ax.set_xlim((0, np.ceil(duration/ms)))
    if bins is not None:
        plt.vlines(bins/ms, ymin=0, ymax=params['N'], colors="#95a5a6", linewidth=1, zorder=0)
    lines = [Line2D([0], [0], color='#8e44ad', lw=2),
             Line2D([0], [0], color='#2c3e50', lw=2)]
    handles = ['exc', 'inh']
    ax.legend(lines, handles, loc="upper right")
    if directory:
        plt.savefig(directory+'/hennequin_reservoir_raster_without_input.pdf', bbox_inches='tight')
        plt.close(fig=fig)

def plot_currents(monitor, figsize=(6, 4), directory=None):
    """
    Plot the some randomly chosen synaptic currents

    Parameters
    ----------
    monitor : StateMonitor
        state monitor of the reservoir
    """
    recorded_synapsis = monitor.record
    #print("Synapse: ", recorded_synapsis)
    fig = plt.figure(figsize=figsize)
    grd = grs.GridSpec(len(recorded_synapsis), 1, wspace=0.0, hspace=0.0)
    for i in range(len(recorded_synapsis)):
        ax = plt.Subplot(fig, grd[i])
        ax.plot(monitor.t/ms, monitor.Ie_syn[i], color='red', linestyle='solid', label="Ie_syn")
        ax.plot(monitor.t/ms, monitor.Ii_syn[i], color='blue', linestyle='dashed', label="Ii_syn")
        fig.add_subplot(ax)
    if directory:
        plt.savefig(directory+'/currents.pdf', bbox_inches='tight')
        plt.close(fig=fig)
    
def plot_currents_distributions(network, figsize=(6, 4), directory=None):
    """
    Plot 

    Parameters
    ----------
    """
    fig = plt.figure(figsize=figsize)
    grid = grs.GridSpec(1, 3, wspace=0.3, hspace=0.0)
    ax1 = plt.Subplot(fig, grid[0])
    ax1.hist(network['gRes'].Itau/pA)
    ax1.set_xlabel("Itau [pA]")
    ax1.set_ylabel("counts")
    ax1.set_yscale("log")
    ax2 = plt.Subplot(fig, grid[1])
    ax2.hist(network['sResRes'].Ie_tau/pA)
    ax2.set_xlabel("Ie_tau [pA]")
    ax2.set_yscale("log")
    ax3 = plt.Subplot(fig, grid[2], sharey=ax2)
    ax3.hist(network['sResRes'].Ii_tau/pA)
    ax3.set_xlabel("Ii_tau [pA]")
    ax3.set_yscale("log")
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    fig.add_subplot(ax3)
    if directory:
        plt.savefig(directory+'/currents_distributions.pdf', bbox_inches='tight')
        plt.close(fig=fig)

def plot_result(X, Y, bins, edges, modulations, snr, figsize=(6, 4), directory=None):
    """
    Plot the readout activity for each sample

    Parameters
    ----------
    X : list (num_samples)
        activity for each sample

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
    """
    num_samples = len(X)
    num_classes = len(np.unique(Y))
    num_samples_per_class = int(num_samples/num_classes)
    for i in range(num_samples):
        sid = i%num_samples_per_class
        fig= plt.figure(figsize=figsize)
        plt.imshow(X[i].T, interpolation='nearest', origin='upper', aspect='auto', \
           extent=[0, bins[0], bins[1], 0], cmap='viridis')
        plt.title('{} @ {} #{}'.format(modulations[Y[i]], snr, sid))
        plt.xlabel('state')
        plt.ylabel('feature')
        if directory:
            plt.savefig(directory+'/{}_{}_{}.pdf'.format(modulations[Y[i]], snr, sid), bbox_inches='tight')
            plt.close(fig=fig)

def plot_readout_states(classes, X, Y, time, figsize=(6, 4), directory=None):
    """
    Plot 

    Parameters
    ----------
    """
    vmin = X.min()
    vmax = X.max()
    fig = plt.figure(figsize=figsize)
    grid = grs.GridSpec(len(classes), 1, wspace=0.0, hspace=0.0)
    for (i, c) in enumerate(classes):
        ax = plt.Subplot(fig, grid[i])
        idx = np.where(Y[:, 2]==c)[0]
        states = [] 
        for x in X[idx]:
            states.append(x[time])
        states = np.array(states)
        im = ax.imshow(states, interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_ylabel(c, labelpad=20, rotation=0, fontsize=18)
        ax.yaxis.tick_right()
        if i<len(classes)-1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("feature", fontsize=18)
        fig.add_subplot(ax)
    ax_cbar = fig.add_axes([1, 0.12, 0.05, 0.76])
    plt.colorbar(im, cax=ax_cbar, orientation='vertical', label='spike count')
    if directory:
        plt.savefig(directory+'/readout_states.pdf', bbox_inches='tight')
        plt.close(fig=fig)

def plot_accuracy(accuracy, figsize=(6, 4), directory=None):
    """
    Plot 

    Parameters
    ----------
    """
    states = np.arange(len(accuracy))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(states, accuracy)
    ax.scatter(states[np.argmax(accuracy)], np.max(accuracy), s=50, marker='o', facecolors='None', edgecolors='red')
    ax.set_ylabel("accuracy", fontsize=18)
    ax.set_xlabel("state", fontsize=18)
    ax.grid(True, linestyle='dashed')
    if directory:
        plt.savefig(directory+'/accuracy.pdf', bbox_inches='tight')
        plt.close(fig=fig)

def plot_similarity(S, Y, labels, figsize=(6, 4), directory=None):
    """
    Plot the cosine similarity matrix between the reservoir 
    readouts of the samples

    Parameters
    ----------
    S : ndarray (num_samples, num_samples)
        similarity matrix

    Y : ndarray (num_samples)
        label of each sample

    labels : list
        names of the classes
    """
    num_samples = S.shape[0]
    num_classes = len(labels)
    num_samples_per_class = int(num_samples/num_classes)
    ticks = [(i+1)*num_samples_per_class for i in range(num_classes)]
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


def plot_network(connectivity, labels, figsize=(6, 4), directory=None):
    """
    Plot the network layers and connections

    Parameters
    ----------
    connectivity : object
        contains the two connectivity matrices as
        i and j indices to be used in the connect method
        of the synapse object in Brian2

    labels : list
        names of the input neurons
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.scatter(connectivity['gen_inp']['i'], connectivity['gen_inp']['j'], c=connectivity['gen_inp']['w'], marker='.')
    ax1.set_xlabel('spike generator')
    ax1.set_ylabel('target neuron')
    ax1.tick_params(direction='in')
    ax1.set_xticklabels(labels)
    ax2.scatter(connectivity['inp_res']['i'], connectivity['inp_res']['j'], c='k', marker='.')
    ax2.set_xlabel('input neuron')
    ax2.tick_params(direction='in')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax2.set_yticklabels([])
    if directory:
        plt.savefig(directory+'/network.pdf', bbox_inches='tight')
        plt.close(fig=fig)

def plot_weights(network, connectivity, N, figsize=(6, 4), directory=None):
    """
    Plot the network weight matrix

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
    """
    W = np.zeros((N, N))
    W[connectivity['res_res']['i'], connectivity['res_res']['j']] = connectivity['res_res']['w']
    w_min = connectivity['res_res']['w'].min()
    w_max = connectivity['res_res']['w'].max()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(W.T, interpolation='nearest', origin='low', aspect='auto', \
                    extent=[0, N, 0, N], cmap='viridis', vmin=w_min, vmax=w_max)
    ax_cbar1 = fig.add_axes([1, 0.1, 0.05, 0.8])
    plt.colorbar(im, cax=ax_cbar1, orientation='vertical', label='weight')
    ax.set_xlabel("source neuron")
    ax.set_ylabel("target neuron")
    if directory:
        plt.savefig(directory+'/weights.pdf', bbox_inches='tight')
        plt.close(fig=fig)

def plot_weights_3D(network, connectivity, N, Ngx, Ngy, Ngz, figsize=(6, 4), directory=None):
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
    """
    source = connectivity['res_res']['i']
    target = connectivity['res_res']['j']
    weight = connectivity['res_res']['w']
    grid = connectivity['grid']
    norm1  = clr.Normalize(vmin=weight.min(), vmax=weight.max())
    smap1 = cmx.ScalarMappable(norm=norm1, cmap='viridis')
    norm2  = clr.Normalize(vmin=0, vmax=9)
    smap2 = cmx.ScalarMappable(norm=norm2, cmap='tab10')
    mx, my, mz = np.meshgrid(np.arange(Ngx), np.arange(Ngy), np.arange(Ngz))
    os.makedirs(directory+'/weights')
    for n in range(N):
        fig = plt.figure(figsize=figsize)
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