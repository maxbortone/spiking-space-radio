from spiking_radio_reservoir import *
from brian2 import mA
import matplotlib.pyplot as plt

# Set brian2 extra compilation arguments
prefs.devices.cpp_standalone.extra_make_args_unix = ["-j6"]

def random_init(connectivity, N, Itau, wGen, wInp):
    # setup input layer
    components = {'generator': None, 'layers': None, 'synapsis': None, 'monitors': None}
    components = setup_input_layer(components, wGen)
    # setup reservoir layer
    components = setup_reservoir_layer(components, connectivity, N, Itau, wInp)
    # set random initial membrane pontential
    components['layers']['gRes'].Imem = np.random.uniform(low=0.5, high=100.0, size=N)*mA
    # add state monitor
    recorded_neurons = np.random.randint(0, high=N, size=10)
    smImem = StateMonitor(components['layers']['gRes'], ['Imem'], name='smImem', record=recorded_neurons)
    components['monitors']['smImem'] = smImem
    # initialize network
    network = TeiliNetwork()
    network.add(components['generator'])
    network.add(list(components['layers'].values()))
    network.add(list(components['synapsis'].values()))
    network.add(list(components['monitors'].values()))
    return network

def test(wGen=3500, wInp=3500, loc_wResE=100, scale_wResE=10, loc_wResI=-50, scale_wResI=5, \
        pIR=0.3, pInh=0.2, AoC=[0.3, 0.2, 0.5, 0.1], DoC=0.5, \
        N=200, tau=20, Ngx=5, Ngy=5, Ngz=8, \
        duration=None):
    start = time.perf_counter()
    print("- running experiment")
    connectivity = setup_schliebs_connectivity(N, pInh, pIR, Ngx, Ngy, Ngz, AoC, DoC, loc_wResE, scale_wResE, loc_wResI, scale_wResI)
    Itau = getTauCurrent(tau*ms)
    # Set C++ backend and time step
    title = 'init_state_{}'.format(os.getpid())
    directory = '../brian2_devices/' + title
    set_device('cpp_standalone', directory=directory, build_on_run=True)
    device.reinit()
    device.activate(directory=directory, build_on_run=True)
    defaultclock.dt = 50*us
    # Initialize network
    network = random_init(connectivity, N, Itau, wGen, wInp)    
    # Run simulation
    network.run(duration, recompile=True)
    # Readout activity
    X, bins, edges = readout(network['mRes'], duration, N, 1, bin_size=1)
    # Plot
    monitor = network['smImem']
    recorded_neurons = monitor.record
    fig, ax = plt.subplots()
    for i in range(len(recorded_neurons)):
        ax.plot(monitor.t/ms, monitor.Imem[i], label="Imem_{}".format(recorded_neurons[i]))
    ax.legend(loc="best")
    plot_result(X, [0], bins, edges, ['random'], 18)
    plot_currents(network['smRes'])
    plot_weights(network, connectivity, N)
    plt.show()
    print("- experiment took {} [s]".format(time.perf_counter()-start))
    return

# Set brian2 extra compilation arguments
prefs.devices.cpp_standalone.extra_make_args_unix = ["-j6"]

# Generate Poisson spike trains

test(duration=1000*ms)