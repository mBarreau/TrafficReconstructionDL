# TrafficReconstructionDL
Code for the paper **Learning-based State Reconstruction for a Scalar Hyperbolic PDE under Lagrangian Sensing** -> [PDF](https://pages.github.com/)

## Description of the files
- godunov.py simulates the problem using a godunov numerical scheme. 
- main.py is the main file, the one to launch.
- neural_network.py is a class for general neural network purposes (using Tensorflow).
- reconstruction_neural_network.py is a class that uses neural_network.py to create a reconstruction neural network.

## Requirements
- Python 3.7.7
- Tensorflow **1.15**
- matplotlib 3.3.1 
- numpy 1.19.1
- [pyDOE 0.3.8](https://pythonhosted.org/pyDOE/)  

## Start a reconstruction
To start the file run main.py.
