# WPI CS 4732 Final Project: N-Body Gravity Simulation

by [Daniel Beckwith](https://github.com/dbeckwith)

[Demo](https://youtu.be/JnmFOf1LoPg)

## About

The goal of this project was to make a fast real-time simulation of gravitational interactions between many particles. The implementation runs a particle simluation on the GPU using an OpenGL compute shader. The shader computes the body-body interactions between every particle and updates their velocities and positions accordingly. Each particle is assigned an initial radius, mass, position, and velocity when the simulation starts based on the given random seed. The particles are arranged to mimic galaxies with a single super-massive star in the center and many smaller stars orbiting the center star. Multiple galaxies interact with each other by colliding and interfering with the orbits of the surrounding stars. The particles are rendered using instanced rendering for the best possible performance, and shaded using the billboard technique to fake a 3D spherical shape. The stars are colored according to their current velocity.

## Installation

First, make sure Python 3.5 is installed. Then, use `pip` to install the dependencies:

```bash
pip install -r requirements.txt
```

This will install the [PyQt5](https://www.riverbankcomputing.com/software/pyqt/download5), [PyOpenGL](http://pyopengl.sourceforge.net/), [pyrr](https://github.com/adamlwgriffiths/Pyrr), and [NumPy](http://www.numpy.org/) packages. If you're on Windows and `pip` can't install some of the packages, you may have to download the appropriate wheel files from [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/) and install them using `pip`:

```bash
pip install path/to/wheel1 path/to/wheel2 ...
```

You may want to do this in a [Virtual Environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/) if you don't have administrative privilages.

## Usage

From the project root, run the python module with the help option to see its usage:

```bash
python -m nbody --help
```
