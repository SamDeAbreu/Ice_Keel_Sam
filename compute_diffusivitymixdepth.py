"""
Computes the diapycnal diffusivity and zmix for all simulations and stores the data as dictionaries in json files. We use Dedalus with the fields being represented in Fourier domains for all of the integration. 

Written by: Sam De Abreu August 2022
"""

#################################
# Imports
import Constants as CON
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import math
import dedalus
import dedalus.public as de
import scipy.signal as sc
import seawater as sw
import json

#################################
# Constants

Nx = CON.Nx
Nz = CON.Nz
H = CON.H
L = CON.L
l = CON.l
z0 = CON.z0
mu = CON.mu

#################################
# Functions

def conv(a_s):
	# Converts string names to numerical values
	if a_s == "200":
		return 2
	elif a_s == "105":
		return 1.5
	elif a_s == "102":
		return 1.2
	elif a_s == "100":
		return 1.0
	elif a_s == "095":
		return 0.95
	else:
		return 0.5

def gen_sigma(a_s):
	# Returns the appropriate float value of the characteristic width for a given height string
	if type(a_s) is str:
		a = conv(a_s)
	else:
		a = a_s
	return 3.9*(H-z0)*a

def sort_rho_z(rho, L_1, L_2, H_1, H_2):
	# Returns the sorted densitiy field (rho_ref[0][0] is the bottom left corner of the domain)
	Nf_x, Ni_x, Nf_z, Ni_z = generate_modes(L_1, L_2, H_1, H_2)
	rho = rho[Ni_x:Nf_x, Ni_z:Nf_z]	  # Restrict rho to selected domain
	rho_sort = np.reshape(-np.sort(-rho.flatten()), (Nf_x-Ni_x, Nf_z-Ni_z), order='F') 
	return rho_sort

def generate_modes(L_1, L_2, H_1, H_2):
	# Generates the grid point index for horziontal and vertical gird spacing Nx and Nz
	Nf_x = math.ceil(Nx/L*L_2)
	Ni_x = math.floor(Nx/L*L_1)
	Ni_z = math.ceil((1-H_2/H)*Nz)
	Nf_z = math.floor((1-H_1/H)*Nz)
	return Nf_x, Ni_x, Nf_z, Ni_z
def create_domain(L_1, L_2, H_1, H_2):
	# Generates a Fourier domain in the specified space
	Nf_x, Ni_x, Nf_z, Ni_z = generate_modes(L_1, L_2, H_1, H_2)
	xbasis = de.Fourier('x', Nf_x-Ni_x, interval=(L_1, L_2))
	zbasis = de.Fourier('z', Nf_z-Ni_z, interval=(H_1, H_2))
	return de.Domain([xbasis, zbasis], grid_dtype=np.float64)

def mixing_depth_calculate(rho, L_1, L_2, ab, cb):
	# Computes the diapycnal diffusivity K and mixing depth zmix
	# Setup fields and variables
	rho_ref = sort_rho_z(rho, L_1, L_2, 0, H)
	rho = rho[:, ::-1] # Flip second axis of rho so rho[0][0] is top left corner of the domain 
	rho_ref = rho_ref[:, ::-1] # Flip second axis of rho_ref so rho_ref[0][0] is top left corner of the domain. This is done to be in accordance with the z array increasing downwards 
	h = (H-z0)*conv(ab) # Maximum keel draft height
	region = 0.5*(1-np.tanh((z-H+h*gen_sigma(ab)**2/(gen_sigma(ab)**2+4*(x-l)**2))/0.01)) # Region mask to avoid the keel
	# Setup domain
	Nf_x, Ni_x, Nf_z, Ni_z = generate_modes(L_1, L_2, 0, H)
	domain = create_domain(L_1, L_2, 0, H)
	x, z = domain.grids(domain.dealias)
	integrand = domain.new_field()
	integrand.set_scales(domain.dealias)
	# Setup of integration fields
	integrand['g'] = region
	A = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0] # Area of integration region
	deriv = 1/np.gradient(rho_ref, z[0], axis=1, edge_order=2)
	rho_z = np.gradient(rho, np.linspace(0, H, Nz), axis=1)[Ni_x:Nf_x]
	rho_x = np.gradient(rho, np.linspace(0, L, Nx), axis=0)[Ni_x:Nf_x]
	nabrho = rho_x**2+rho_z**2
	nabrho[nabrho < 1e-6] = 0 # Set possible salinity leakage to 0 to prevent interference in calculating zmix (see Section 2.3 in the paper)
	integrand['g'] = 9.8*mu*region*deriv*nabrho/(sw.dens0(28,-2)) # Integrand of phi
	phi_dval = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]
	# Compute zmix
	phi_dvalx = de.operators.integrate(integrand, 'x').evaluate()['g'][0] # Compute the z-slice integral of phi for finding zmix.
	z_f = H # Maximum zmix by default
	for z_m in np.linspace(0, H, Nz)[3::3]:
		height_lim = 0.5*(1-np.tanh((z[0]-z_m)/0.01)) # Height mask such that we only integrate over z < z_m
		integrand['g'] = height_lim*phi_dvalx 
		phi_dtemp = de.operators.integrate(integrand, 'z').evaluate()['g'][0][0]
		if phi_dtemp/phi_dval > 0.9:
			z_f = z_m
			break
	# Compute the diapycnal diffusivity
	N_sq = 9.8/sw.dens0(28,-2) * np.average(np.gradient(np.average(rho_ref, axis=0), z[0]))
	K = phi_dval/(N_sq*A)
	return (float(K), float(z_f))

def create_K_json(L_1, L_2):
	a_s = ['a005', 'a095', 'a102', 'a200']
	c_s = ['c005', 'c100', 'c105', 'c200']
	times = [220, 260, 450, 450] # Run times in file count
	K = {}
	z_mix = {}
	for i in range(len(a_s)):
		for j in range(len(c_s)):
			K_temp = 0
			z_mix_temp = 0
			l = 0
			for k in range(70, times[j], 3): # Loop through all files
				with h5py.File('new/data-mixingsim-{0}{1}-00/data-mixingsim-{0}{1}-00_s{2}.h5'.format(a_s[i], c_s[j], k), mode='r') as f:
					temp = mixing_depth_calculate(f['tasks']['rho'][0], L_1, L_2, a_s[i], c_s[j])
					K_temp += temp[0]
					z_mix_temp += temp[1] 
					l += 1
			# Compute average quantities
			K[a_s[i]+c_s[j]] = K_temp/l
			z_mix[a_s[i]+c_s[j]] = z_mix_temp/l
			print(i,j)
	# Store data in json formatted as dictionaries 
	json.dump(K, open('K_values_{0}-{1}.txt'.format(L_1, L_2), 'w'))
	json.dump(z_mix, open('z_mix_values_{0}-{1}.txt'.format(L_1, L_2), 'w'))

def generate_new_set():
	# Generates diapycnal diffusivitiy and zmix json for upstream and downstream
	print('Building upstream K and zmix json')
	create_K_json(160, l)
	print('Building downstream K and zmix json')
	create_K_json(l, L-40)
	

# Run
if __name__ == "__main__":
	generate_new_set()

