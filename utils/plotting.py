import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grs
from matplotlib.lines import Line2D
from brian2 import ms, nA, pA

def plot_sample(signal, time_stim, indices, times, title=None, figsize=(6, 4)):
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

def plot_sample_with_labels(signal, time_stim, indices, times, labels, time_labels, bins, title=None, figsize=(6, 4)):
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
    
def plot_sample_with_reconstruction(signal, reconstruction, time_stim, indices, times, title=None, figsize=(6, 4)):
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
    
def plot_input_synapsis(stateMon, spikeMon, weight, title):
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
    
def plot_input_layer(stateMon, spikeMon, title):
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
    
def plot_input_layer_with_membrane_potential(indices, times, network, duration, bins=None):
    fig = plt.figure(figsize=(12, 9))
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
    if bins!=None:
        ax1.vlines(bins/ms, ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], colors="#95a5a6", linewidth=1, zorder=0)
        ax2.vlines(bins/ms, ymin=ax2.get_ylim()[0], ymax=ax2.get_ylim()[1], colors="#95a5a6", linewidth=1, zorder=0)
        ax3.vlines(bins/ms, ymin=ax3.get_ylim()[0], ymax=ax3.get_ylim()[1], colors="#95a5a6", linewidth=1, zorder=0)
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    fig.add_subplot(ax3)
    
def plot_hennequin_reservoir_raster(connectivity, params, network, times, duration, bins=None):
    input_neurons_Iup = connectivity['inp_res']['j'][connectivity['inp_res']['i']==0]
    input_neurons_Idn = connectivity['inp_res']['j'][connectivity['inp_res']['i']==1]
    input_neurons_Qup = connectivity['inp_res']['j'][connectivity['inp_res']['i']==2]
    input_neurons_Qdn = connectivity['inp_res']['j'][connectivity['inp_res']['i']==3]
    M = params['Ngx']*params['Ngy']
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
        elif n < M:
            colors.append('#8e44ad')
        else:
            colors.append('#2c3e50')
    positions = []
    for n in range(params['N']):
        idx = np.where(network['mRes'].i==n)[0]
        positions.append(network['mRes'].t[idx]/ms)
    plt.figure(figsize=(12, 9))
    plt.eventplot(positions, colors=colors)
    plt.axvline(times[-1]/ms, linestyle='dashed', color='#95a5a6')
    plt.gca().invert_yaxis()
    plt.ylabel("neuron index")
    plt.xlabel("time [ms]")
    plt.xlim((0, np.ceil(duration/ms)))
    if bins!=None:
        plt.vlines(bins/ms, ymin=0, ymax=params['N'], colors="#95a5a6", linewidth=1, zorder=0)
    lines = [Line2D([0], [0], color='#2ecc71', lw=2),
             Line2D([0], [0], color='#e74c3c', lw=2),
             Line2D([0], [0], color='#e67e22', lw=2),
             Line2D([0], [0], color='#3498db', lw=2),
             Line2D([0], [0], color='#8e44ad', lw=2),
             Line2D([0], [0], color='#2c3e50', lw=2)]
    handles = ['I.up', 'I.dn', 'Q.up', 'Q.dn', 'exc', 'inh']
    plt.legend(lines, handles, loc="upper right")

def plot_hennequin_reservoir_raster_without_input(connectivity, params, network, times, duration, bins=None):
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
    plt.figure(figsize=(12, 9))
    plt.eventplot(positions, colors=colors)
    plt.axvline(times[-1]/ms, linestyle='dashed', color='#95a5a6')
    plt.gca().invert_yaxis()
    plt.ylabel("neuron index")
    plt.xlabel("time [ms]")
    plt.xlim((0, np.ceil(duration/ms)))
    if bins!=None:
        plt.vlines(bins/ms, ymin=0, ymax=params['N'], colors="#95a5a6", linewidth=1, zorder=0)
    lines = [Line2D([0], [0], color='#8e44ad', lw=2),
             Line2D([0], [0], color='#2c3e50', lw=2)]
    handles = ['exc', 'inh']
    plt.legend(lines, handles, loc="upper right")
    
def plot_currents_distributions(currents):
    fig = plt.figure(figsize=(12, 9))
    grid = grs.GridSpec(1, 3, wspace=0.2, hspace=0.0)
    ax1 = plt.Subplot(fig, grid[0])
    ax1.hist(currents['gRes']['Itau']/pA)
    ax1.set_xlabel("Itau [pA]")
    ax1.set_ylabel("counts")
    ax1.set_title("Neurons")
    ax2 = plt.Subplot(fig, grid[1])
    ax2.hist(currents['sResRes']['Ie_tau']/pA)
    ax2.set_xlabel("Ie_tau [pA]")
    ax2.set_title("Excitatory synapsis")
    ax3 = plt.Subplot(fig, grid[2], sharey=ax2)
    ax3.hist(currents['sResRes']['Ii_tau']/pA)
    ax3.set_xlabel("Ii_tau [pA]")
    ax3.set_title("Inhibitory synapsis")
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    fig.add_subplot(ax3)