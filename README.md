# polytope_roll
Robotic manipulation to roll a polytope using trajectory optimization and controls.

## Installation instructions
We recommend making a python virtual environment.

```
cd /desired/location/
git clone https://github.com/ebianchi/polytope_roll.git
cd polytope_roll
python3.10 -m venv venv
venv/bin/pip install --upgrade pip
source venv/bin/activate
```
At this point, ensure that `/desired/location/polytope_roll` is included in the `$PYTHONPATH` environment variable.  Then requirements can be installed via two steps.  First:
```
pip install polytope_roll
```
This installs all of the bundled requirements, including, at the time of creation of this repository, `imageio`, `matplotlib`, `numpy`, and `sympy`.  There is one remaining requirement that, in our experience, has only worked properly if directly pip installing the git repository url.
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
