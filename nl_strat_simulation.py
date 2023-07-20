'''
Simulates sigmoidally stratified flow under a Versoria-shaped ice keel with a phase field method using Dedalus. The code is originally based off of Hester et al. (2021).  

Can be run in parallel (x40)

Written by: Rosalie Cormier, August 2021
Edited by: Sam De Abreu, April 2022

BSD 3-Clause License

Copyright (c) 2023, SamDeAbreu

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
# Imports
import dedalus
import dedalus.public as de
from dedalus.extras import flow_tools, plot_tools
from dedalus.tools import post
from dedalus.core.operators import GeneralFunction
import numpy as np
import time
import sys
import h5py
import seawater as sw
from mpi4py import MPI
import logging
import os
from os.path import join
import shutil

logger = logging.getLogger()
################################
# Functions

def sigmoid(x, a=1):
	return 0.5 * (np.tanh(x/a) + 1)

def versoria_keel(x, depth, centre, stddev, H):
	return H - depth*stddev**2/(stddev**2+4*(x-centre)**2)

def stratification(z, scale, z0, Delta, b):
	return scale * np.tanh((z-z0) / Delta) + b

################################
# Parameters
Fr = 2
eta = 2

# Stratification parameters
T_w = -2 # Temperature of the water (C)
scale = -1 # Scale of salinity difference between the mixed layers (psu)
z0 = 72 # Summer mixed layer depth controller parameter (NOTE: the actual depth is H-z0) (m)
b = 1e-1 # Pycnocline scale (m)
S = 29 # Salinity Constant (ML salinities given by S +- scale) (psu)
rho0 = sw.dens0(S+scale, T_w) # Reference density (and density of summer mixed layer) (kg/m^3)
rho1 = rho0
rho2 = sw.dens0(S-scale, T_w)

# Dimensional  
C_w = 1 # Salinity modifier for EOS (g/kg)
nu = 2e-3 # Viscosity (momentum diffusivity) (m^2/s)
mu = 2e-3 # Salt mass diffusivity (m^2/s)
L, H = 960, 80 # Domain length and height (m)
l, h = 600, eta * (H-z0) # Location of center of keel and maximum keel draft (m)
sigma = 3.9 * h # Characteristic keel width (m)
U = Fr * np.sqrt((H-z0) * 9.8 * (rho2-rho1)/rho1) # Fixed flow velocity (m/s)

# Dimensionless
Re = 1 / nu # Reynolds number
Sc = nu / mu # Schmidt number (should be ~1)

# Phase field parameters
delta = 5e-3 # Regulates the salinity equation within the keel 
epsilon = 0.125 # Phase field steepness parameter (m)
xi = (2/5) * 1e-3 * Re * (epsilon * 4 / 2.648228)**2 # Damping and restoring time scale in phase field (s). Partially taken from Hester et al. (2021)

# Simulation parameters
Nx, Nz = 1280, 640 # Number of grid points in horizontal and vertical
dt = 6e-3 # Time step (s)
steps = 1050000 # Total number of steps to run. Simulation time = steps * dt

# Save parameters
sim_name = 'mixingsim-Fr20eta20' # Name to save simulation by
restart_sim = True # True = restart simulation
file_handler_mode = 'overwrite' 
save_freq = 150 # How often to write to file
save_max = 15 # Maximum number of writes per file
print_freq = 1000 # Frequency for printing diagnostic messages
wall_time = 60*60*23 # Maximum wall time allowed
save_dir = '.' # Directory to save to

###################################

#Bases and domain

xbasis = de.Fourier('x', Nx, interval=(0, L), dealias=3/2)
zbasis = de.SinCos('z', Nz, interval=(0, H), dealias=3/2)
domain = de.Domain([xbasis, zbasis], grid_dtype=np.float64)

x, z = domain.grids(domain.dealias)
kx, kz = domain.elements(0), domain.elements(1)

###################################
# Fields

#Define GeneralFunction subclass
class ParityFunction(GeneralFunction):

	def __init__(self, domain, layout, func, args=[], kw={}, out=None, parity={},):
		super().__init__(domain, layout, func, args=[], kw={}, out=None,)
		self._parities = parity

	def meta_parity(self, axis):
		return self._parities.get(axis, 1) #Even by default


# Temperature field
T = domain.new_field()
T.set_scales(domain.dealias)
T.meta['z']['parity'] = +1 
T['g'] = 1 # Constant temperature everywhere

# Buoyancy field
def buoyancy_func(T, C):
	return -9.8 * (sw.dens0(C_w * C['g'], T_w * T['g']) - rho0) / rho0

B = ParityFunction(domain, layout='g', func=buoyancy_func,)

# Wall term
def accel_func(solver):
	# Multiplier to the wall term to simulate the keel acceleration
	t = solver.sim_time
	# The acceleration works by scaling the magnitude of the wall term. Wall term smaller in magnitude -> Velocity is restored to a value less than U
	# We perform a "linear acceleration" for t < 1800s~80t_0. Due to what we think is the nonlinearities in the equations, the velocity field reaches 98% of U within 900s~42t_0 (not linearly). That is, the actual acceleration is over 900s with a ~2% increase in U over the next 900s. We consider this negligible and that the actual acceleration takes place over the first 900s.  
	# The scaling factor of 5e5 is introduced to bring the acceleration within the claimed time window above (found via trial and error)
	accel_f = np.piecewise(t, [t <= 0, 0<t<=1800, t > 1800], [0, lambda t: t/500000, 1])
	return accel_f

accel = ParityFunction(domain, layout='g', func=accel_func,)

wall = domain.new_field() # Wall field (psi in paper)
wall.set_scales(domain.dealias)
wall.meta['z']['parity'] = 1 
wall['g'] = (sigmoid(-(x-0.02*L), a=8*epsilon) + sigmoid(x-0.98*L, a=8*epsilon))
wall['c'] *= np.exp(-kx**2 / 5e6) # Spectral smoothing

# Buoyancy multiplier for parity constraints
par = domain.new_field()
par.set_scales(domain.dealias)
par.meta['z']['parity'] = -1 
par['g'] = np.tanh(-(z-H) / 0.05) * np.tanh(z / 0.05)
par['c'] *= np.exp(-kx**2 / 5e6) # Spectral smoothing

# Keel mask, 1 when inside the keel and 0 outside
keel_mask = domain.new_field()
keel_mask.set_scales(domain.dealias)
keel_mask.meta['z']['parity'] = +1 
keel_mask['g'] = sigmoid(z-versoria_keel(x, h, l, sigma, H)+0.5, a=2*epsilon)

# Stratification to be used in wall and keel (should match initial condition for salinity)
strat = domain.new_field()
strat.set_scales(domain.dealias)
strat.meta['z']['parity'] = +1
strat['g'] = stratification(z, scale, z0, delta, b)*(1-keel_mask['g']) + (S+scale)*keel_mask['g']

# Upstream regime masking
prof1 = domain.new_field()
prof1.set_scales(domain.dealias)
prof1.meta['z']['parity'] = +1
prof1['g'] = 0.5*(-np.tanh((x-l)/0.00125)+1)

# Downstream regime masking
prof2 = domain.new_field()
prof2.set_scales(domain.dealias)
prof2.meta['z']['parity'] = +1
prof2['g'] = 0.5*(np.tanh((x-l)/0.00125)+1)

# Density field
def dens_func(T, C):
	return sw.dens0(C_w * C['g'], T_w * T['g'])

rho = ParityFunction(domain, layout='g', func=dens_func,)

###################################
# Setup of equations of motion

mixing = de.IVP(domain, variables=['u', 'w', 'C', 'p', 'phi', 'ct']) # p refers to gauge pressure

# Set parities
mixing.meta['u', 'p', 'C', 'phi', 'ct']['z']['parity'] = 1
mixing.meta['w']['z']['parity'] = -1

# Define parameters 
params = [Nx, Nz, wall, delta, epsilon, mu, xi, h, U, L, H, B, T, par, nu, scale, z0, b, S, strat, rho, rho0, prof1, prof2, l, keel_mask, accel]
param_names = ['Nx', 'Nz', 'wall', 'delta', 'epsilon', 'mu', 'xi', 'h', 'U', 'L', 'H', 'B', 'T', 'par', 'nu', 'scale', 'z0', 'b', 'S', 'strat', 'rho', 'rho0', 'prof1', 'prof2', 'l', "keel_mask", "accel"]

for param, name in zip(params, param_names):
	mixing.parameters[name] = param

mixing.substitutions['q'] = 'dz(u) - dx(w)' # Define vorticity

# Equations of motion
mixing.add_equation('dx(u) + dz(w) = 0', condition='(nx != 0) or (nz != 0)')
	#Eq 3e (Hester)
mixing.add_equation('p = 0', condition='(nx == 0) and (nz == 0)')
mixing.add_equation('dt(u) + dx(p) - nu*dz(q) = -w*q - (phi/xi)*u - (wall*accel/xi)*(u-U)')
	#Eq 3d (Hester)
mixing.add_equation('dt(w) + dz(p) + nu*dx(q) = u*q - (phi/xi)*w + par*B - (wall*accel/xi)*w')
	#Eq 3d (Hester)
mixing.add_equation('dt(C) - mu*(dx(dx(C)) + dz(dz(C))) = -(u*dx(C) + w*dz(C)) - mu*(dx(C)*dx(phi)+dz(C)*dz(phi))/(1-phi+delta) - (phi/xi)*(C-strat) - (wall*accel/xi)*(C-strat)')
	#Eq 3c (Hester)
mixing.add_equation('dt(phi) = 0')
mixing.add_equation('dt(C) - ct = 0')

# Build solver
solver = mixing.build_solver(de.timesteppers.SBDF3)
logger.info('Solver built')

####################################
# Set initial conditions

u, w, C, p, phi, ct = variables = [solver.state[field] for field in mixing.variables]

for field in variables:
	field.set_scales(domain.dealias)

# Configure arguments for parity functions
B.original_args = B.args = [T, C]
rho.original_args = rho.args = [T, C]
accel.original_args = accel.args = [solver]

if restart_sim:
	# Restart simulation
	w['g'] = 0 # Initial vertical velocity
	p['g'] = 0 # Initial gauge pressure
	phi['g'] = sigmoid(z-versoria_keel(x, h, l, sigma, H)-1, a=2*epsilon) # Keel phase field
	#f['g'] = 0 #Use this alternative to run simulation without a keel
	u['g'] = U * (1 - phi['g']) # Initial velocity field
	C['g'] = stratification(z, scale, z0, b, S)*(1 - phi['g']) + (S+scale)*phi['g'] # Initial stratification
	ct['g'] = 0 
	solver.stop_iteration = steps
	file_handler_mode = 'overwrite'
else:
	# Start from a checkpoint
	write, initial_timestep = solver.load_state('checkpoints-{0}/checkpoints-{0}_s3.h5'.format(sim_name)) # Manually set the checkpoint file
	solver.stop_iteration = 2*steps # Set how many more steps to complete
	file_handler_mode = 'append' # Don't clear the files already present in directory

# Save configurations
solver.stop_wall_time = wall_time
solver.stop_sim_time = np.inf

# Save state variables
analysis = solver.evaluator.add_file_handler(join(save_dir, 'data-{}-{:0>2d}'.format(sim_name, 0)), iter=save_freq, max_writes=save_max, mode=file_handler_mode)
analysis.add_system(solver.state)

# Save other values
analysis.add_task('rho', layout='g', name='rho') # Density field
analysis.add_task("1/(integ(prof1*(1-keel_mask), 'x'))*integ(C*prof1*(1-keel_mask), 'x')", layout='g', name='avg_salt_prof1') # Average salt in Upstream
analysis.add_task("1/(integ(prof2*(1-keel_mask), 'x'))*integ(C*prof2*(1-keel_mask), 'x')", layout='g', name='avg_salt_prof2') # Average salt in Downstream
analysis.add_task("q", layout='g', name='vorticity') # Vorticity field

# Checkpoints
checkpoints = solver.evaluator.add_file_handler('checkpoints-{0}'.format(sim_name), sim_dt=(steps-1)*dt/2, max_writes=1, mode='append')
checkpoints.add_system(solver.state)

# Save other parameters (uncomment to save all stored parameters)
#parameters = solver.evaluator.add_file_handler(join(save_dir, 'parameters-{}-{:0>2d}'.format(sim_name, restart)), iter=save_freq, max_writes=save_max, mode=file_handler_mode)

#for task in mixing.variables:
#	parameters.add_task(task)

#for name in param_names:
#	parameters.add_task(name)

#parameters.add_task('q')



######################################
# Main loop

start_time = time.time()
while solver.proceed:
	if solver.iteration % print_freq == 0:
 		# Print simulation diagnostic and check for numerical instabilities
		maxspeed = u['g'].max()
		logger.info('{:0>6d}, u max {:f}, dt {:.5f}, time {:.2f}, sim time {:.5f}'.format(solver.iteration, maxspeed, dt, (time.time()-start_time)/60, dt*solver.iteration))
		if np.isnan(maxspeed):
			break

	solver.step(dt)
solver.step(dt)
