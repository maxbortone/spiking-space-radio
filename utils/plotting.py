import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grs
from brian2 import ms

def plot_sample(signal, time_stim, indices, times, title):
    fig = plt.figure()
    grid = grs.GridSpec(3, 1, wspace=0.0, hspace=0.0)
    ax1 = plt.Subplot(fig, grid[0])
    ax1.set_title(title)
    ax1.plot(time_stim/ms, signal[0, :], color='#2ecc71', label='I')
    ax1.plot(time_stim/ms, signal[1, :], color='#e67e22', label='Q')
    ax1.legend(loc='upper right')
    ax2 = plt.Subplot(fig, grid[1])
    ax2.set_yticks([])
    ax2.set_ylabel('I')
    ax2.vlines(times[indices==0]/ms, ymin=0.0, ymax=1.0, colors='#e74c3c', linewidth=1)
    ax2.vlines(times[indices==1]/ms, ymin=-1.0, ymax=0.0, colors='#3498db', linewidth=1)
    ax3 = plt.Subplot(fig, grid[2])
    ax3.set_yticks([])
    ax3.set_ylabel('Q')
    ax3.set_xlabel(r'Time [ms]')
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
    ax4.set_xlabel('Time [ms]')
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
    ax2.set_xlabel('Time [ms]')
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)