from setuptools import setup

setup(
   name='polytope_roll',
   version='1.0',
   description=f'Robotic manipulation to roll a polytope using trajectory ' \
               + f'optimization and controls.',
   author='Bibit Bianchini',
   author_email='bibit@seas.upenn.edu',
   packages=['polytope_roll'],
   install_requires=[
      'imageio',
      'matplotlib',
      'numpy',
      'sympy'
   ],
)