# Classification of radio signals on a neuromorphic chip in space

This repository contains the source code for my master thesis project conducted at the Institute of Neuroinformatics in the fall semester of 2018. A reservoir computing approach was taken to implement a recurrent neural network for automatic radio signal modulation recognition on neuromorphic hardware for space applications. The simulations are developed using the [Brian2](https://brian2.readthedocs.io/en/stable/) simulator and the [Teili](https://teili.readthedocs.io/en/latest/) neuromorphic toolkit to mimic actual hardware as close as possible.

## Organization

The repository is structured as follows:

* **notebooks**: Juypter notebooks documenting various steps of my research
* **thesis**: PDF version of the thesis and the presentation
* **utils**: library of various methods for data preprocessing, plotting and reservoir computing developed during the project
* `spiking_radio_reservoir.py`: main code base for the simulations
* `bayesian_optimization_process.py`: hyperparameter tuning for the reservoir