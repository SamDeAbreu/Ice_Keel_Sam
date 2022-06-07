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
import math
import dedalus
import dedalus.public as de
import scipy.signal as sc
import seawater as sw

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
sigma = CON.sigma
E_0 = (9.8*(sw.dens0(30,-2)-sw.dens0(28,-2))*z0**4) #initial potential energy at z=24
phi_0 = sw.dens0(28,-2)*np.sqrt(9.8*(sw.dens0(30,-2)-sw.dens0(28,-2))/sw.dens0(28,-2)*z0**(9))

xbasis = de.Fourier('x', 1228, interval=(10, L-10))
zbasis = de.SinCos('z', 640, interval=(0, H))
domain = de.Domain([xbasis, zbasis], grid_dtype=np.float64)

x, z = domain.grids(domain.dealias)
#z = (np.arange(Nz)/Nz)*H #use equally spaced grid points since SinCos is equally spaced. This doesn't affect integ


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
		rho = f['tasks']['rho'][i][26:1254]
		#rho = np.array([[6,1,2], [5, 9, 10], [1, 7, 2], [1, 2, 8]])
		#z = np.array([0, 10, 20])
		#print(f['tasks']['rho'].dims[0]['sim_time'][:])
		#print(f['tasks']['rho'].dims[2][0][:])
		#z_sort alg: 
		# 1) Sort density values and use the same indices to sort a list A=[0,1,2,...]
		# 2) Create a new B=[0,1,2,...] list parallel to the new sorted density values (new z location of sorted density values)
		# 3) Sort B by the indices to sort A 
		# 4) Compute the z-values for each index in B. Now B[mx][mz] represents the reference z value of the sorted density for a density at (mx,mz) 
		ind = z[0][np.argsort(np.argsort(-rho, axis=None))//1228]
		z_sort_height = np.reshape(ind, (1228, 640))
		rho_sort = np.reshape(-np.sort(-rho.flatten()), (1228,640), order='F')
	return rho_sort, z_sort_height
def compute_mixedlayerdepth(h5_file):
	with h5py.File(h5_file, mode='r') as f:
		data = f['tasks']['rho'][0]
		posz = []
		low = math.ceil((CON.l+CON.sigma)*CON.Nx/CON.L)
		for mx in range(low, CON.Nx):
			rho_min = data[mx][-1]
			diff = data[mx] - rho_min
			i = min(np.argwhere(diff <= 0.1))[0]
			posz.append(-z[0][Nz-i])
	return np.mean(posz), posz
def compute_potential_energies(h5_file, i):
	rho_ref, z_ref = sort_rho_z(h5_file, i)
	with h5py.File(h5_file, mode='r') as f:
		rho = f['tasks']['rho'][i][26:1254]
		w = f['tasks']['w'][i][26:1254]

		integrand = domain.new_field()
		integrand.set_scales(domain.dealias)
		integrand.meta['z']['parity'] = +1 #Parity effects on the integrated results are minimal

		integrand['g'] = 9.8*rho*z_ref
		E_b = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]
	
		integrand['g'] = 9.8*rho*(z-z_ref)
		E_a = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]

		integrand['g'] = 9.8*rho*z
		E_p = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]
		
		
		deriv = np.resize(np.diff(z_ref)/np.diff(rho), (1228, 640))+np.resize(np.diff(z_ref, axis=0)/np.diff(rho, axis=0), (1228,640))
		
		integrand['g'] = rho
		rho_z = de.operators.differentiate(integrand, 'z').evaluate()['g']
		rho_x = de.operators.differentiate(integrand, 'x').evaluate()['g']
		

		integrand['g'] = -9.8*mu*np.nan_to_num(deriv, posinf=0, neginf=0)*(rho_z**2+rho_x**2)
		phi_d = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]

		integrand['g'] = 9.8*w*rho
		phi_z = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]

		rho_top = np.mean(rho, axis=0)[-1]
		rho_bot = np.mean(rho, axis=0)[0]
		phi_i = -9.8*mu*L*(rho_top-rho_bot)
		
		print(E_b, E_a, E_p, phi_d, phi_z, phi_i, phi_d-phi_i)
		return E_b, E_a, E_p, phi_d, phi_z
def plot_potential_energies():
	data = np.loadtxt('potentialdata_{0}_{1}-{2}.txt'.format(fileno, splice0, splice1), skiprows=1, unpack=True)
	time = data[0]
	Eb = data[1]
	Ea = data[2]
	Ep = data[3]
	phi_d = data[4]
	phi_z = data[5]
	ked_rate = data[6]
	MLD = data[7]

	phi_z_fft = list(abs(np.fft.fft(phi_z)))
	phi_z_fft[phi_z_fft.index(max(phi_z_fft))] = 0
	phi_z_fft = np.array(phi_z_fft)
	peaks = sc.find_peaks(phi_z_fft, height=0.8e8)
	
	freq = np.fft.fftfreq(len(time))
	for i in range(len(peaks[0])):
		print("Height: {0} freq: {1}".format(peaks[1]['peak_heights'][i], freq[peaks[0][i]]))
	
	fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex='col')

	ax1.plot(time, Eb/E_0, label="Background potential energy")
	ax1.set_ylabel("$E_b/E$")
	ax1.set_title('Dimensionless Energies and Vertical Bouyancy flux ($h={0}$m, U=${1}$m/s)'.format(h, CON.U))
	ax1.grid()
	#plt.savefig('bgpotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	#plt.clf()

	ax3.plot(time, Ea/E_0, label="Available potential energy")
	#plt.xlabel("Simulation time (s)")
	ax3.set_ylabel("$E_a/E$")
	ax2.grid()
	#plt.savefig('avpotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	#plt.clf()

	ax2.plot(time, phi_z/phi_0, label="Rate of mixing")
	#plt.xlabel("Simulation time (s)")
	ax2.set_ylabel("$\\phi_z/\\phi$")
	#plt.savefig('phizpotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	#plt.clf()
	ax3.set_xlabel("Simulation time (s)")
	plt.tight_layout()
	plt.grid()
	plt.savefig('avbgphiz_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))

	plt.plot(time, Ep/E_0, label="Total potential energy")
	plt.xlabel("Simulation time (s)")
	plt.ylabel("$E_{{tot}}/E_{{tot_{{i}}}}$")
	plt.title("Total Potential Energy (h = "+str(h)+")")
	plt.savefig('totalpotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf()

	plt.plot(time[40:], phi_d[40:]/phi_0, label="Rate of mixing")
	plt.xlabel("Simulation time (s)")
	plt.ylabel("$\\phi_d/\\phi_{{d_{{i}}}}$")
	plt.title("Rate of mixing ($h={0}$m, U=${1}$m/s)".format(h, CON.U))
	plt.grid()
	plt.savefig('phidpotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf()

	plt.plot(time, MLD)
	plt.xlabel('Simulation Time (s)')
	plt.ylabel("Average Mixed Layer Depth")
	plt.title("Time Series of Average MLD")
	plt.grid()
	plt.savefig('MLDpotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf()

	"""plt.plot(time, phi_i/abs(phi_i[0]), label="Rate of mixing")
	plt.xlabel("Simulation time (s)")
	plt.ylabel("$\\phi_i/\\phi_{{i_{{i}}}}$")
	plt.title("Rate of internal energy to $E_b$ (h = "+str(h+4)+")")
	plt.savefig('phiipotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf()

	plt.plot(time, (phi_d-phi_i)/abs(phi_d[0]-phi_i[0]), label="Rate of mixing")
	plt.xlabel("Simulation time (s)")
	plt.ylabel("$(\\phi_d-phi_i)/(\\phi_{{d_{{i}}}}-phi_{{i_{{i}}}})$")
	plt.title("Rate of Available energy to $E_b$ (h = "+str(h+4)+")")
	plt.savefig('phid-phiipotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf()"""

	plt.plot(time, ked_rate, label="Rate of mixing")
	plt.xlabel("Simulation time (s)")
	plt.ylabel("$\\epsilon/\\epsilon_{{i}}$")
	plt.title("Rate of Kinetic Energy Dissipation ($h={0}$m, U=${1}$m/s)".format(h, CON.U))
	plt.grid()
	plt.savefig('ked_rateenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf() 

	plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(phi_z_fft)))
	plt.xlim(-0.1,0.1)
	plt.xlabel("Frequency (Hz)")
	plt.ylabel("Absolute value of FFT")
	plt.savefig('test.png')
	plt.clf()
	
def create_data_file():
	h5_files = sort_h5files(files, splice0, splice1)
	file = open('potentialdata_{0}_{1}-{2}.txt'.format(fileno, splice0, splice1), 'a+')
	data = [[], [], [], [], [], [], [], [], []]
	for h5file in h5_files[::3]:
		with h5py.File(h5file, mode='r') as f: 
			times = f['tasks']['u'].dims[0]['sim_time'][:]
			print(h5file)
			energies = compute_potential_energies(h5file, 0)
			ML = compute_mixedlayerdepth(h5file)
			data[0].append(times[0])
			data[1].append(energies[0])
			data[2].append(energies[1])
			data[3].append(energies[2])
			data[4].append(energies[3])
			data[5].append(energies[4])
			data[6].append(f['tasks']['ked_rate'][0][0][0])
			data[7].append(ML[0])
			data[8].append(ML[1])
	MLD_std = np.std(data[8])
	print('Writing to file')
	file.write(str(MLD_std)+'\n')
	for i in range(len(data[0])):
		file.write(str(data[0][i])+'\t'+str(data[1][i])+'\t'+str(data[2][i])+'\t'+str(data[2][i])+'\t'+str(data[3][i])+'\t'+str(data[4][i])+'\t'+str(data[5][i])+'\t' +str(data[6][i])+'\t'+str(data[7][i])+'\n')
	file.close()
	print('Done')
create_data_file()
#plot_potential_energies()
