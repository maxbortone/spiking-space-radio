import numpy as np
from brian2 import *
from teili import TeiliNetwork
from teili.core.groups import Neurons, Connections
from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn


class SpikingRadioReservoir():
    
    def __init__(self):
        self.network = None
        self.generator = None
        self.layers = {}
        self.synapsis = {}
        self.monitors = {}
        self.connectivity = {}
        self.indices = np.array([])
        self.times = np.array([])*ms
        self.title = ''
    
    def set_input_layer(self, num_neurons=2, refractory_period=0.0, weight=1.0):
        self.generator = SpikeGeneratorGroup(4, self.indices, self.times, name='gGen')
        gInp = Neurons(num_neurons, equation_builder=DPI(num_inputs=2), refractory=refractory_period*ms, name='gInp')
        gInp.Iahp = 0.5*pA
        self.layers.update({'gInp': gInp})
        sGenInp = Connections(self.generator, gInp, equation_builder=DPISyn(), method='euler', name='sGenInp')
        sGenInp.connect(i=[0, 1, 2, 3], j=[0, 0, 1, 1])
        sGenInp.weight = weight
        self.synapsis.update({'sGenInp': sGenInp})
        mGen = SpikeMonitor(self.generator, name='mGen')
        mInp = SpikeMonitor(gInp, name='mInp')
        smInp = StateMonitor(gInp, ('Iin'), record=True, name='smInp')
        smSyn = StateMonitor(sGenInp, variables=["Ii_syn", "Ie_syn"], record=True, name="smSyn")
        self.monitors.update({'mGen': mGen, 'mInp': mInp, 'smInp': smInp, 'smSyn': smSyn})
        
    def set_reservoir_layer(self, num_neurons=10, refractory_period=0.0, p_rr=0.3, p_ir=0.3, \
                            weight_rr=1.0,  weight_ir=1.0):
        self.num_neurons = num_neurons
        gRes = Neurons(num_neurons, equation_builder=DPI(num_inputs=4), refractory=refractory_period*ms, name='gRes')
        gRes.Iahp = 0.5*pA
        self.layers.update({'gRes': gRes})
        sResRes = Connections(gRes, gRes, equation_builder=DPISyn(), method='euler', name='sResRes')
        sResRes.connect(p=p_rr)
        sResRes.weight = weight_rr
        sInpRes = Connections(self.layers['gInp'], gRes, equation_builder=DPISyn(), method='euler', name='sInpRes')
        sInpRes.connect(p=p_ir)
        sInpRes.weight = weight_ir
        self.synapsis.update({'sInpRes': sInpRes, 'sResRes': sResRes})
        mRes = SpikeMonitor(gRes, name='mRes')
        smRes = StateMonitor(gRes, ('Iin'), record=True, name='smRes')
        self.monitors.update({'mRes': mRes, 'smRes': smRes})
    
    def set_spikes(self, indices, times):
        self.generator.set_spikes(indices, times)
    
    def add(self):
        self.network = TeiliNetwork()
        self.network.add(self.generator)
        self.network.add(list(self.layers.values()))
        self.network.add(list(self.synapsis.values()))
        self.network.add(list(self.monitors.values()))
        
    def run(self, duration, title):
        self.network.run(duration)
        self.title = title
        
    def store(self, name):
        self.network.store(name)
        
    def restore(self, name):
        self.network.restore(name)
    
    def plot_network(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.scatter(self.network['sGenInp'].i, self.network['sGenInp'].j, c='k', marker='.')
        ax1.set_xlabel('Source neuron index')
        ax1.set_ylabel('Target neuron index')
        ax1.set_xticks([0, 1, 2, 3])
        ax1.set_xticklabels(['I.up', 'I.dn', 'Q.up', 'Q.dn'])
        ax1.set_yticks([0, 1])
        ax1.set_title('Generator')
        ax2.scatter(self.network['sInpRes'].i, self.network['sInpRes'].j, c='k', marker='.')
        ax2.set_xlabel('Source neuron index')
        #ax2.set_ylabel('Target neuron index')
        ax2.set_xticks([0, 1])
        ax2.set_title('Input')
        C = np.zeros((self.num_neurons,self.num_neurons))
        C[self.network['sResRes'].i, self.network['sResRes'].j] = 1.0
        ax3.imshow(C, aspect='auto', origin='lower')
        ax3.set_xlabel('Source neuron index')
        #ax3.set_ylabel('Target neuron index')
        ax3.set_title('Reservoir')
        plt.show()
        
    def plot_activity(self):
        plt.figure()
        input_neurons_I = self.network['sInpRes'].j[np.where(self.network['sInpRes'].i==0)[0]]
        input_neurons_Q = self.network['sInpRes'].j[np.where(self.network['sInpRes'].i==1)[0]]
        reservoir_neurons = np.unique(self.network['mRes'].i) 
        for k in reservoir_neurons:
            spikes = self.network['mRes'].i == k
            if k in input_neurons_I:
                color = '#3498db'
            elif k in input_neurons_Q:
                color = '#e74c3c'
            else:
                color = 'k'
            spikes = self.network['mRes'].i == k
            plt.scatter(self.network['mRes'].t[spikes]/ms, self.network['mRes'].i[spikes], \
                        c=color, s=2, marker=',')
        spikes = self.network['mInp'].i == 0
        plt.scatter(self.network['mInp'].t[spikes]/ms, 1.0-self.network['mInp'].i[spikes]+self.num_neurons, \
                    c='#2ecc71', s=2, marker=',')
        spikes = self.network['mInp'].i == 1
        plt.scatter(self.network['mInp'].t[spikes]/ms, 1.0-self.network['mInp'].i[spikes]+self.num_neurons, \
                    c='#e67e22', s=2, marker=',')
        plt.xlabel('Time [ms]')
        plt.ylabel('Reservoir neuron index')
        plt.title(self.title)
        
        

def getTau(I):
    '''
    Compute DPI time constant tau in seconds for a given current value:
    Cmem : DPI capacitance [F]
    Ut : Thermal voltage [V]
    I : Current, must be given in [A]
    tau : (Cp*Ut)/(kappa*I) [sec]
    '''
    Cmem = 1.5*pfarad
    Ut = 25*mV
    kappa = 0.705
    return (Cmem*Ut)/(kappa*I)

def getTauCurrent(tau):
    '''
    Compute the current in Amperes necessary to get a desired time constant:
    tau : Time constant, must be given in [sec]
    Cmem : DPI capacitance [F]
    Ut : Thermal voltage [V]
    I : (Cp*Ut)/(kappa*tau) [A]
    '''
    Cmem = 1.5*pfarad
    Ut = 25*mV
    kappa = 0.705
    if tau == 0:
        return 2.390625e-05
    return (Cmem*Ut)/(kappa*tau)

def getAhpTau(I):
    '''
    Compute adaptation time constant tau in seconds for a given current value:
    Cahp : Adaptation capacitance [F]
    Ut : Thermal voltage [V]
    Itauahp : Current, must be given in [A]
    tauahp : (Cahp*Ut)/(kappa*Itauahp) [sec]
    '''
    Cahp = 1.0*pfarad
    Ut = 25*mV
    kappa = 0.705
    return (Cahp*Ut)/(kappa*I)

def getAhpTauCurrent(tau):
    '''
    Compute the current in Amperes necessary to get a desired time constant:
    tauahp : Adaptation time constant, must be given in [sec]
    Cahp : DPI capacitance [F]
    Ut : Thermal voltage [V]
    Itauahp : (Cahp*Ut)/(kappa*tauahp) [A]
    '''
    Cahp = 1.0*pfarad
    Ut = 25*mV
    kappa = 0.705
    if tau == 0:
        return 2.390625e-05
    return (Cahp*Ut)/(kappa*tau)
