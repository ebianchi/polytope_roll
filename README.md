# polytope_roll
Robotic manipulation to roll a polytope using trajectory optimization and controls.

## Installation instructions
We recommend making a python virtual environment.  Once navigating into a desired directory, clone the repository and set up the virtual environment via:
```
git clone https://github.com/ebianchi/polytope_roll.git
cd polytope_roll
python3 -m venv venv
venv/bin/pip install --upgrade pip
source venv/bin/activate
```
At this point, ensure that `/desired/location/polytope_roll` is included in the `$PYTHONPATH` environment variable.  Without changing directory, you can do this immediately after the previous step via:
```
export PYTHONPATH=$PYTHONPATH:$PWD
```

Then requirements can be installed via three steps.  First, `pip install` the requirements in the current directory via:
```
pip install .
```
This installs all of the bundled requirements, including, at the time of creation of this repository, `imageio`, `matplotlib`, `numpy`, and `sympy`.  However this also installs `polytope_roll` as a package, which will be problematic if you want to edit the source code in this repository.  You can remove the `polytope_roll` package while keeping its requirements via:
```
pip uninstall polytope_roll
```
Lastly, there is one remaining requirement that, in our experience, has only worked properly if directly pip installing the git repository url.
```
pip install git+https://github.com/AndyLamperski/lemkelcp.git
```
This installs [AndyLamperski](https://github.com/AndyLamperski/lemkelcp)'s python implementation of the Lemke algorithm for solving linear complementarity problems (LCPs), which is used to run the 2-dimensional toy simulation via [Stewart and Trinkle, 1996](https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-0207(19960815)39:15%3C2673::AID-NME972%3E3.0.CO;2-I?casa_token=iTtnVw6eUTQAAAAA:vukRTwhhqfZxVWFulF4LowUc2Bcui8V_FM7Lo9D1N8CGQ0Iitz9c7cKA8owKFLczTRLuSfiXE9Ake5k).

## Run example
You can run an example of a 2-dimensional polytope falling under gravity and colliding inelastically with the ground via:
```
python toy_2d/src/simulate_polytope.py
```
This call should generate a gif of the falling polytope, saving it into `polytope_roll/toy_2d/out/`.

## Debugging
In case the example does not run, try ensuring the installed packages match the following versions:
```
$ pip list
Package         Version
--------------- -------
contourpy       1.0.5
cycler          0.11.0
fonttools       4.38.0
imageio         2.22.2
kiwisolver      1.4.4
lemkelcp        0.1
matplotlib      3.6.1
mpmath          1.2.1
numpy           1.23.4
packaging       21.3
Pillow          9.2.0
pip             22.3
pyparsing       3.0.9
python-dateutil 2.8.2
setuptools      65.4.1
six             1.16.0
sympy           1.11.1
```
