'''
Written by: Rosalie Cormier, August 2021

Mixing simulation with nonlinear initial stratification

Can be run in parallel (x40)
'''

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
logger = logging.getLogger()

d = de.operators.differentiate

import os
from os.path import join
import shutil

################################

def sigmoid(x, a=1):
	return 0.5 * (np.tanh(x/a) + 1)

def gaussian_keel(x, depth, centre, stddev, H):
	return H - depth * np.exp(-((x-centre)**2)/(2*stddev**2))

def stratification(z, scale, z0, Delta, b):
	return scale * np.tanh((z-z0) / Delta) + b

################################

#Dimensional parameters
U = 0.03 #m/s
T_w = 1 #C
c_p = 4.2 #J/gC
L_T = 3.342 #J/g
C_w = 1 #g/kg
nu = 5e-4 #m^2/s #Viscosity (momentum diffusivity)
kappa = 1.3e-3 #cm^2/s
mu = 5e-4 #m^2/s #Salt mass diffusivity
m = 0.056 #C/(g/kg)
L, H = 110, 15 #m
l, h = 70, 7 #m

Re = 1 / nu
Sc = nu / mu #Should be close to 1
S = L_T / (c_p * T_w)
delta = 5e-3
epsilon = 0.125 #Two gridboxes
beta = 4/2.648228 #Not optimized
eta = 1e-5 * Re * (beta * epsilon)**2

#Parameters defining stratification
scale = -1
z0 = 10.56
Delta = 1e-1
b = 27

#Save parameters
Nx, Nz = 256, 128
dt = 5e-4 #s #For certain speeds and mixed-layer depths, you can increase this by a factor of 10 to reduce runtime

sim_name = 'mixingsimTest'
restart = 0 #Integer

steps = 1000 #At 800000 steps, takes many hours to run
save_freq = 30 #15
save_max = 15
print_freq = 1000 #Decrease this for diagnostic purposes if the code isn't working
wall_time = 60*60*23
save_dir = '.'

###################################

#Bases and domain
xbasis = de.Fourier('x', Nx, interval=(0, L), dealias=3/2)
zbasis = de.SinCos('z', Nz, interval=(0, H), dealias=3/2)
domain = de.Domain([xbasis, zbasis], grid_dtype=np.float64)

x, z = domain.grids(domain.dealias)
kx, kz = domain.elements(0), domain.elements(1)

#Wall penalty boundary
wall = domain.new_field()
wall.set_scales(domain.dealias)
wall.meta['z']['parity'] = 1
wall['g'] = sigmoid(-(x - 0.02*L), a=4*epsilon) + sigmoid(x - 0.98*L, a=4*epsilon)
wall['c'] *= np.exp(-kx**2 / 5e6) #Spectral smoothing

#Define GeneralFunction subclass
class ParityFunction(GeneralFunction):

	def __init__(self, domain, layout, func, args=[], kw={}, out=None, parity={},):
		super().__init__(domain, layout, func, args=[], kw={}, out=None,)
		self._parities = parity

	def meta_parity(self, axis):
		return self._parities.get(axis, 1) #Even by default

rho0 = sw.dens0(0, 0)

T = domain.new_field()
T.set_scales(domain.dealias)
T.meta['z']['parity'] = +1
T['g'] = -2 #Approx. freezing temperature (C)

def buoyancy_func(T, C):
	#return -9.8*0.77*(C['g']-35)
	return -9.8 * 100 * (sw.dens0(C_w * C['g'], T_w * T['g']) - rho0) / rho0

B = ParityFunction(domain, layout='g', func=buoyancy_func,)

#Buoyancy multiplier for parity constraints
par = domain.new_field()
par.set_scales(domain.dealias)
par.meta['z']['parity'] = -1
par['g'] = np.tanh(-(z-H) / 0.05) * np.tanh(z / 0.05)
par['c'] *= np.exp(-kx**2 / 5e6) #Spectral smoothing

#Stratification to be used in wall (should match initial condition for salinity)
strat = domain.new_field()
strat.set_scales(domain.dealias)
strat.meta['z']['parity'] = +1
strat['g'] = scale * np.tanh((z-z0) / Delta) + b

def dens_func(T, C):
	return sw.dens0(C_w * C['g'], T_w * T['g'])
	#return 1025*(0.77*(C['g'])-35) 

rho = ParityFunction(domain, layout='g', func=dens_func,)

###################################

#Mixing problem
mixing = de.IVP(domain, variables=['u', 'w', 'C', 'p', 'f', 'ct'])
	#p refers to gauge pressure
mixing.meta['u', 'p', 'C', 'f', 'ct']['z']['parity'] = +1
mixing.meta['w']['z']['parity'] = -1

params = [Nx, Nz, delta, epsilon, mu, eta, h, U, L, H, B, T, par, S, nu, wall, scale, z0, Delta, b, strat, rho, rho0]
param_names = ['Nx', 'Nz', 'delta', 'epsilon', 'mu', 'eta', 'h', 'U', 'L', 'H', 'B', 'T', 'par', 'S', 'nu', 'wall', 'scale', 'z0', 'Delta', 'b', 'strat', 'rho', 'rho0']

for param, name in zip(params, param_names):
	mixing.parameters[name] = param

mixing.substitutions['q'] = 'dz(u) - dx(w)' #Vorticity

mixing.add_equation('dx(u) + dz(w) = 0', condition='(nx != 0) or (nz != 0)')
	#Eq 3e (Hester)
mixing.add_equation('p = 0', condition='(nx == 0) and (nz == 0)')
mixing.add_equation('dt(u) + dx(p) - nu*dz(q) = -w*q - (f/eta)*u - (wall/eta)*(u-U)')
	#Eq 3d (Hester)
mixing.add_equation('dt(w) + dz(p) + nu*dx(q) = u*q - (f/eta)*w + par*B - (wall/eta)*w')
	#Eq 3d (Hester)
mixing.add_equation('dt(C) - mu*(dx(dx(C)) + dz(dz(C))) = -(u*dx(C) + w*dz(C)) - mu*(dx(C)*dx(f)+dz(C)*dz(f))/(1-f+delta) - (f/eta)*(C-strat) - (wall/eta)*(C-strat)')
	#Eq 3c (Hester)
mixing.add_equation('dt(f) = 0')
mixing.add_equation('dt(C) - ct = 0')

#Build solver
solver = mixing.build_solver('SBDF2')
logger.info('Solver built')

####################################

#Set initial conditions
u, w, C, p, f, ct = variables = [solver.state[field] for field in mixing.variables]

for field in variables:
	field.set_scales(domain.dealias)

B.original_args = B.args = [T, C]
rho.original_args = rho.args = [T, C]

w['g'] = 0
p['g'] = 0
f['g'] = sigmoid(z-gaussian_keel(x, h, l/2, 6, H), a=2*epsilon)
	#The stddev is variable! We set its value in the above line -- might be worth defining a new variable for it
#f['g'] = 0 #Use this alternative to run simulation without a keel
u['g'] = U * (1 - f['g'])
C['g'] = stratification(z, scale, z0, Delta, b) #* (1 - f['g']) + 5*f['g']
ct['g'] = 0

#Save configurations
solver.stop_iteration = steps
solver.stop_wall_time = wall_time
solver.stop_sim_time = np.inf

#Save state variables
analysis = solver.evaluator.add_file_handler(join(save_dir, 'data-{}-{:0>2d}'.format(sim_name, restart)), iter=save_freq, max_writes=save_max, mode='overwrite')
analysis.add_system(solver.state)

#Save other values
analysis.add_task("1/(L-(-36*log((H-z)/h))**(1/2))*integ(C, 'x')", layout='g', name='average_salt')
analysis.add_task("integ(T - S*f, 'x', 'z')", layout='g', name='energy')
analysis.add_task("integ((1-f)*C, 'x', 'z')", layout='g', name='salt')
analysis.add_task("q", layout='g', name='vorticity')
#analysis.add_task("B", layout='g', name='B')
analysis.add_task("integ(2*(nu/rho)*((dx(u))**2+(dz(u))**2+(dx(w))**2+(dz(w))**2), 'x', 'z')", layout='g', name='ked_rate')
	#Kinetic energy dissipation rate
#analysis.add_task("w*C", layout='g', name='tsf')
	#Turbulent salinity flux
#analysis.add_task("mu*dx(C)", layout='g', name='dsfx')
	#Diffusive salinity flux, x-comp.
#analysis.add_task("mu*dz(C)", layout='g', name='dsfz')
	#Diffusive salinity flux, z-comp.

#Save parameters
parameters = solver.evaluator.add_file_handler(join(save_dir, 'parameters-{}-{:0>2d}'.format(sim_name, restart)), iter=save_freq, max_writes=save_max, mode='overwrite')

for task in mixing.variables:
	parameters.add_task(task)

for name in param_names:
	parameters.add_task(name)

parameters.add_task('q')

######################################

#Main loop
start_time = time.time()
while solver.proceed:

	if solver.iteration % print_freq == 0:
		maxspeed = u['g'].max()
		logger.info('{:0>6d}, u max {:f}, dt {:.5f}, time {:.2f}, sim time {:.5f}'.format(solver.iteration, maxspeed, dt, (time.time()-start_time)/60, dt*solver.iteration))
		if np.isnan(maxspeed):
			break

	solver.step(dt)
solver.step(dt)
