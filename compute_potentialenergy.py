"""
Usage:
	mp4_plots.py <files>... [--fileno=<str>] [--splice=<str>]

Written by: Sam De Abreu August 2021
"""
import Constants as CON
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import dedalus
import dedalus.public as de


from docopt import docopt
args = docopt(__doc__)
files = args['<files>'] #=h5 file location for analysis
fileno = args['--fileno'] #Name of outputted mp4s
splice = args['--splice'] #Specified files for analysis
splice0 = int(splice[:splice.find('-')])
splice1 = int(splice[splice.find('-')+1:])

Nx = CON.Nx
Nz = CON.Nz
H = CON.H
L = CON.L
l = CON.l
h = CON.h
z0 = CON.z0
h_z = CON.H-z0
mu = CON.mu
sigma = 6

xbasis = de.Chebyshev('x', 582-58, interval=(10, L-10))
zbasis = de.Chebyshev('z', 640, interval=(0, H))
domain = de.Domain([xbasis, zbasis], grid_dtype=np.float64)

x, z = domain.grids()
wall_mask = 0.5*(np.tanh((x-10)/0.025)+1)-0.5*(np.tanh((x-100)/0.025)+1)

def sort_h5files(h5_files, splice0, splice1):
	#Sort h5 files
	temp_list = []
	temp_list2 = []
	for filename in h5_files:
		temp_list.append(int(filename[filename.find("_")+2:filename.find(".")]))
	for i in range(splice0, splice1+1):
		temp_list2.append(h5_files[temp_list.index(i)])
	h5_files = temp_list2
	return h5_files
def sort_rho_z(h5_file, i):
	with h5py.File(h5_file, mode='r') as f:
		rho = f['tasks']['rho'][i][58:582]
		#rho = np.array([[2,3, 5, 67, 89, 70, 90], [4,1, 28, 211, 234, 123, 322], [7,3,89, 90, 901, 120, 111], [0,65,45, 34, 12, 902, 963]])
		#print(f['tasks']['rho'].dims[0]['sim_time'][:])
		#print(f['tasks']['rho'].dims[2][0][:])
		#z_sort alg: 
		# 1) Sort density values and use the same indices to sort a list A=[0,1,2,...]
		# 2) Create a new B=[0,1,2,...] list parallel to the new sorted density values (new z location of sorted density values)
		# 3) Sort B by the indices to sort A 
		# 4) Compute the z-values for each index in B. Now B[mx][mz] represents the reference z value of the sorted density for a density at (mx,mz) 
		ind = np.argsort(rho.flatten(), axis=0) 
		#top_sort = np.sort(rho.flatten()) 
		bot_sort = np.arange(524*640)[ind] 		
		ind2 = np.argsort(bot_sort)
		#Sorted such that (0,-H) is (0,0). Done this way so the magnitude of height for something at a deeper depth is smaller
		#Essentially measuring height from the bottom of the domain
		z_sort_height = H*((Nz-1)-np.reshape(np.arange(524*640)[ind2]//524, (524,640)))/640
		rho_sort = np.reshape(-np.sort(-rho.flatten()), (524,640), order='F')
	return rho_sort, z_sort_height
def compute_potential_energies(h5_file, i):
	rho_ref, z_ref = sort_rho_z(h5_file, i)
	with h5py.File(h5_file, mode='r') as f:
		rho = f['tasks']['rho'][i][58:582]
		nab_rho_sq = f['tasks']['nabla_rho_sq'][i][58:582]
		w = f['tasks']['w'][i][58:582]

		integrand = domain.new_field()
		integrand2 = domain.new_field()
		integrand3 = domain.new_field()
		integrand4 = domain.new_field()
		integrand5 = domain.new_field()
		integrand6 = domain.new_field()
	
		integrand['g'] = 9.8*rho*z_ref
		E_b = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]
	
		integrand2['g'] = 9.8*rho*(z-z_ref)
		E_a = de.operators.integrate(integrand2, 'x', 'z').evaluate()['g'][0][0]

		integrand3['g'] = 9.8*rho*z
		E_p = de.operators.integrate(integrand3, 'x', 'z').evaluate()['g'][0][0]
		
		deriv = np.resize(np.diff(z_ref)/np.diff(rho), (524, 640))
		integrand4['g'] = -9.8*mu*deriv*abs(nab_rho_sq)
		phi_d = de.operators.integrate(integrand4, 'x', 'z').evaluate()['g'][0][0]

		integrand5['g'] = -9.8*w*rho
		phi_z = de.operators.integrate(integrand5, 'x', 'z').evaluate()['g'][0][0]

		integrand6['g'] = 1/(110-20)*rho
		rho_avg = de.operators.integrate(integrand6, 'x').evaluate()['g'][0]
		phi_i = -9.8*mu*(110-20)*(rho_avg[-1]-rho_avg[0])

		return E_b, E_a, E_p, phi_d, phi_z, phi_i
def plot_potential_energies():
	data = np.loadtxt('potentialenergy_{0}_{1}-{2}.txt'.format(fileno, splice0, splice1), unpack=True)
	time = data[0]
	Eb = data[1]
	Ea = data[2]
	Ep = data[3]
	phi_d = data[4]
	phi_z = data[5]
	phi_i = data[6]
	r = phi_d-phi_i
	plt.plot(time, Eb, label="Background potential energy")
	plt.xlabel("Simulation time (s)")
	plt.ylabel("Background Potential energy (J)")
	plt.title("Background Potential Energy (h = "+str(h)+")")
	plt.savefig('bgpotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf()

	plt.plot(time, Ea, label="Available potential energy")
	plt.xlabel("Simulation time (s)")
	plt.ylabel("Available Potential energy (J)")
	plt.title("Available Potential Energy (h = "+str(h)+")")
	plt.savefig('avpotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf()

	plt.plot(time, Ep, label="Total potential energy")
	plt.xlabel("Simulation time (s)")
	plt.ylabel("Total Potential energy (J)")
	plt.title("Total Potential Energy (h = "+str(h)+")")
	plt.savefig('totpotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf()

	plt.plot(time, phi_d, label="Rate of mixing")
	plt.xlabel("Simulation time (s)")
	plt.ylabel("$\\phi_d$")
	plt.title("Rate of mixing (h = "+str(h)+")")
	plt.savefig('phidpotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf()

	plt.plot(time, phi_z, label="Rate of mixing")
	plt.xlabel("Simulation time (s)")
	plt.ylabel("$\\phi_z$")
	plt.title("Vertical Bouyancy (h = "+str(h)+")")
	plt.savefig('phizpotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf()

	plt.plot(time, phi_i, label="Rate of mixing")
	plt.xlabel("Simulation time (s)")
	plt.ylabel("$\\phi_i$")
	plt.title("Rate of internal energy to $E_b$ (h = "+str(h)+")")
	plt.savefig('phiipotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf()

	plt.plot(time, phi_d-phi_i, label="Rate of mixing")
	plt.xlabel("Simulation time (s)")
	plt.ylabel("$\\phi_d-phi_i$")
	plt.title("Rate of Available energy to $E_b$ (h = "+str(h)+")")
	plt.savefig('phid-ipotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	

    
def create_data_file():
	h5_files = sort_h5files(files, splice0, splice1)
	file = open('potentialenergy_{0}_{1}-{2}.txt'.format(fileno, splice0, splice1), 'a+')
	for h5file in h5_files[::2]:
		with h5py.File(h5file, mode='r') as f: 
			times = f['tasks']['u'].dims[0]['sim_time'][:]
			print(h5file)
			energies = compute_potential_energies(h5file, 0)
			file.write(str(times[0])+'\t'+str(energies[0])+'\t'+str(energies[1])+'\t'+str(energies[2])+'\t'+str(energies[3])+'\t'+str(energies[4])+'\t'+str(energies[5]) + '\n')
	file.close()
create_data_file()
#plot_potential_energies()
