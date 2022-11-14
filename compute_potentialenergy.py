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
import json

from docopt import docopt
args = docopt(__doc__)
files = args['<files>'] #=h5 file location for analysis
fileno = args['--fileno'] #Name of outputted mp4s
splice = args['--splice'] #Specified files for analysis
splice0 = int(splice[:splice.find('-')]) - 70
splice1 = int(splice[splice.find('-')+1:])

def conv(a_s):
	if a_s == "200":
		return 2
	elif a_s == "102":
		return 1.2
	elif a_s == "095":
		return 0.95
	else:
		return 0.5
def gen_sigma(a_s):
	if type(a_s) is str:
		a = conv(a_s)
	else:
		a = a_s
	return 3.9*(H-z0)*a

Nx = CON.Nx
Nz = CON.Nz
H = CON.H
L = CON.L
l = CON.l
a_s = fileno[fileno.find("-a")+2:fileno.find("-a")+5]
c_s = fileno[fileno.find("c")+1:fileno.find("c")+4]
a = conv(a_s)
print(a)
z0 = CON.z0
h = a*(H-z0)
b = CON.b
Delta = CON.Delta
h_z = CON.H-z0
mu = CON.mu
nu = CON.nu
sigma = 3.9*h
DB = 9.8*(sw.dens0(30,-2)-sw.dens0(28,-2))/sw.dens0(28,-2)
E_0 = (9.8*(sw.dens0(30,-2)-sw.dens0(28,-2))*(H-z0)**4) 
phi_0 = sw.dens0(28,-2)*np.sqrt(DB**3*(H-z0)**(7))
t_0 = np.sqrt((H-z0)/DB)



def sort_h5files(h5_files, splice0, splice1):
	#Sort h5 files
	temp = []
	for filename in h5_files:
		if 'h5' in filename:
			temp.append(filename)
	return sorted(temp, key=lambda x: int(x[x.find("_")+2:x.find(".")]))[splice0:splice1+1]
def sort_rho_z(rho, L_1, L_2, H_1, H_2):
	domain = create_domain(L_1, L_2, H_1, H_2)
	x, z = domain.grids(domain.dealias)
	Nf_x, Ni_x, Nf_z, Ni_z = generate_modes(L_1, L_2, H_1, H_2)
	rho = rho[Ni_x:Nf_x, Ni_z:Nf_z]	 
	#ind = z[0][np.argsort(np.argsort(-rho, axis=None))//(Nf_x-Ni_x)]
	#z_sort_height = np.reshape(ind, (Nf_x-Ni_x, Nf_z-Ni_z))
	rho_sort = np.reshape(-np.sort(-rho.flatten()), (Nf_x-Ni_x, Nf_z-Ni_z), order='F')
	return rho_sort, 0
def compute_mixedlayerdepth(h5_file, L_1, L_2, H_1, H_2):
	domain = create_domain(L_1, L_2, H_1, H_2)
	x, z = domain.grids(domain.dealias)
	with h5py.File(h5_file, mode='r') as f:
		data = f['tasks']['rho'][0]
		posz = []
		low = math.ceil(L_1*CON.Nx/CON.L)
		high = math.floor(L_2*CON.Nx/CON.L)
		for mx in range(low, high):
			rho_min = data[mx][-1]
			diff = data[mx] - rho_min
			i = min(np.argwhere(diff <= 0.8))[0]
			posz.append(-z[0][Nz-i])
	return np.mean(posz), posz
def compute_potential_energies(h5_file, L_1, L_2, H_1, H_2):
	domain = create_domain(L_1, L_2, H_1, H_2)
	x, z = domain.grids(domain.dealias)
	#region = 1/32*(np.tanh((x-L_1)/0.01)+1)*(1-np.tanh((x-L_2)/0.01))*(-np.tanh((z-(H-H_1))/0.01)+1)*(1+np.tanh((z-(H-H_2))/0.01))*(1-np.tanh((z-H+h*sigma**2/(sigma**2+4*(x-l)**2))/0.01))
	region = 0.5*(1-np.tanh((z-H+h*sigma**2/(sigma**2+4*(x-l)**2))/0.01))
	Nf_x, Ni_x, Nf_z, Ni_z = generate_modes(L_1, L_2, H_1, H_2)
	with h5py.File(h5_file, mode='r') as f:
		rho_ref, z_ref = sort_rho_z(f['tasks']['rho'][0], L_1, L_2, H_1, H_2)
		rho = f['tasks']['rho'][0][Ni_x:Nf_x, Ni_z:Nf_z]
		w = f['tasks']['w'][0][Ni_x:Nf_x, Ni_z:Nf_z]
		u = f['tasks']['u'][0][Ni_x:Nf_x, Ni_z:Nf_z]

		integrand = domain.new_field()
		integrand.set_scales(domain.dealias)
		
		integrand['g'] = region
		A = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]
		integrand['g'] = 9.8*rho*z_ref*region
		E_b = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]
	
		integrand['g'] = 9.8*rho*(z-z_ref)*region
		E_a = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]

		integrand['g'] = 9.8*rho*z*region
		E_p = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]

		#deriv = np.resize(np.diff(z_ref)/np.diff(rho), (1228, 640))+np.resize(np.diff(z_ref, axis=0)/np.diff(rho, axis=0), (1228,640))
		deriv = 1/np.gradient(rho_ref, z[0], axis=1, edge_order=2)
		#deriv = np.gradient(z_ref, z[0], axis=1, edge_order=2)/np.gradient(rho, z[0], axis=1, edge_order=2)+np.gradient(z_ref, np.concatenate(x).ravel(), axis=0, edge_order=2)/np.gradient(rho, np.concatenate(x).ravel(), axis=0, edge_order=2)
		rho_z = np.gradient(rho, z[0], axis=1, edge_order=2)
	
		rho_x = np.gradient(rho, np.concatenate(x).ravel(), axis=0, edge_order=2)
		integrand['g'] = -9.8*region*mu*np.nan_to_num(deriv, nan=0, posinf=0, neginf=0)*(rho_z**2+rho_x**2)/(A*sw.dens0(28,-2))
		phi_d = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]
		rho_avg_s = np.average(np.gradient(rho_ref, z[0], axis=1, edge_order=2), axis=0)
		N_sq = -9.8/sw.dens0(28,-2)*np.average(rho_avg_s)
		K = phi_d/N_sq
			
		integrand['g'] = 9.8*w*rho*region
		phi_z = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]

		u_x = np.gradient(u, np.concatenate(x).ravel(), axis=0, edge_order=2)
		u_z = np.gradient(u, z[0], axis=1, edge_order=2)
		w_x = np.gradient(w, np.concatenate(x).ravel(), axis=0, edge_order=2)
		w_z = np.gradient(w, z[0], axis=1, edge_order=2)
		integrand['g'] = nu*(2*u_x**2+(u_z+w_x)**2+2*w_z**2)/A
		ked = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]

		L_t = np.sqrt(np.average((z-z_ref)**2))
		
		integrand['g'] = 0.5*region*rho*(u_x**2+u_z**2+w_x**2+w_z**2)
		E_k = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]
		return {'E_b': E_b, 'E_a': E_a, 'E_p': E_p, 'phi_d': phi_d, 'phi_z': phi_z, 'ked': ked, 'L_t': L_t, 'E_k': E_k, 'K': K}
def plot_potential_energies():
	data = np.loadtxt('potentialdata_{0}_{1}-{2}.txt'.format(fileno, splice0, splice1), skiprows=2, unpack=True)
	time = data[0]
	Eb = data[1]
	Ea = data[2]
	Ep = data[3]
	phi_d = data[4]
	phi_z = data[5]
	ked_rate = data[6]
	MLD = data[7]
	m = data[8]
	E_k = data[9]
	create_figs(time, Eb, Ea, Ep, phi_d, phi_z, ked_rate, MLD, m, 'Total', E_k=E_k)
	Eb = data[10]
	Ea = data[11]
	Ep = data[12]
	phi_d = data[13]
	phi_z = data[14]
	ked_rate = data[15]
	m = data[16]
	create_figs(time, Eb, Ea, Ep, phi_d, phi_z, ked_rate, MLD, m, 'Downstream')
	Eb = data[17]
	Ea = data[18]
	Ep = data[19]
	phi_d = data[20]
	phi_z = data[21]
	ked_rate = data[22]
	m = data[23]
	create_figs(time, Eb, Ea, Ep, phi_d, phi_z, ked_rate, MLD, m, 'Upstream')
def create_figs(time, Eb, Ea, Ep, phi_d, phi_z, ked_rate, MLD, m, type, E_k=None):
	phi_z_fft = list(abs(np.fft.fft(phi_z)))
	phi_z_fft[phi_z_fft.index(max(phi_z_fft))] = 0
	phi_z_fft = np.array(phi_z_fft)
	peaks = sc.find_peaks(phi_z_fft, height=0.8e8)
	
	freq = np.fft.fftfreq(len(time))
	for i in range(len(peaks[0])):
		print("Height: {0} freq: {1}".format(peaks[1]['peak_heights'][i], freq[peaks[0][i]]))
	
	fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex='col')

	ax1.plot(time/t_0, Eb/E_0, label="Background potential energy")
	ax1.set_ylabel("$E_b/E$")
	ax1.set_title('Dimensionless Energies and Vertical Bouyancy flux ($h={0}$m, U=${1}$m/s)'.format(h, CON.U))
	ax1.grid()

	ax3.plot(time/t_0, Ea/E_0, label="Available potential energy")
	ax3.set_ylabel("$E_a/E$")
	ax2.grid()

	ax2.plot(time/t_0, phi_z/phi_0, label="Rate of mixing")
	ax2.set_ylabel("$\\phi_z/\\phi$")
	ax3.set_xlabel("$t/\\sqrt{{z_0/\\Delta B}}$")
	plt.tight_layout()
	plt.grid()
	plt.savefig(type+'_avbgphiz_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))

	plt.clf()
	plt.plot(time/t_0, Ep/E_0, label="Total potential energy")
	plt.xlabel("$t/\\sqrt{{z_0/\\Delta B}}$")
	plt.ylabel("$E_{{tot}}/E_{{tot_{{i}}}}$")
	plt.title("Total Potential Energy (h = "+str(h)+")")
	plt.savefig(type+'_totalpotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf()

	plt.plot(time/t_0, phi_d/phi_0, label="Rate of mixing")
	plt.xlabel("$t/\\sqrt{{z_0/\\Delta B}}$")
	plt.ylabel("$\\phi_d/\\phi_{{d_{{i}}}}$")
	plt.title("Rate of mixing ($h={0}$m, U=${1}$m/s)".format(h, CON.U))
	plt.grid()
	plt.savefig(type+'_phidpotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf()

	plt.plot(time/t_0, MLD)
	plt.xlabel("$t/\\sqrt{{z_0/\\Delta B}}$")
	plt.ylabel("Average Mixed Layer Depth")
	plt.title("Time Series of Average MLD")
	plt.grid()
	plt.savefig(type+'_MLDpotentialenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf()


	plt.plot(time/t_0, ked_rate, label="Rate of mixing")
	plt.xlabel("$t/\\sqrt{{z_0/\\Delta B}}$")
	plt.ylabel("$\\epsilon/\\epsilon_{{i}}$")
	plt.title("Rate of Kinetic Energy Dissipation ($h={0}$m, U=${1}$m/s)".format(h, CON.U))
	plt.grid()
	plt.savefig(type+'_ked_rateenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf() 

	plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(phi_z_fft)))
	plt.xlim(-0.1,0.1)
	plt.xlabel("Frequency (Hz)")
	plt.ylabel("Absolute value of FFT")
	plt.savefig(type+'_test.png')
	plt.clf()

	plt.plot(time/t_0, Eb/m-Eb[0]/m[0])
	plt.xlabel("$t/\\sqrt{{z_0/\\Delta B}}$")
	plt.ylabel("$\\Delta E_b$ (J/kg)")
	plt.savefig(type+'_thrope.png_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
	plt.clf()

	if E_k is not None:
		plt.plot(time/t_0, E_k/E_0, label="Total potential energy")
		plt.xlabel("$t/\\sqrt{{z_0/\\Delta B}}$")
		plt.ylabel("$E_{{k}}/E_{{k_{{i}}}}$")
		plt.title("Total Kinetic Energy (h = "+str(h)+")")
		plt.savefig(type+'_totalkineticenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
		plt.clf()

		plt.plot(time/t_0, Ep/E_0+E_k/E_0, label="Total potential energy")
		plt.xlabel("$t/\\sqrt{{z_0/\\Delta B}}$")
		plt.ylabel("$E/E_{{k_{{i}}}}$")
		plt.title("Total Energy (h = "+str(h)+")")
		plt.savefig(type+'_totalenergy_{0}_{1}-{2}.png'.format(fileno, splice0, splice1))
		plt.clf()

def generate_modes(L_1, L_2, H_1, H_2):
	Nf_x = math.ceil(Nx/L*L_2)
	Ni_x = math.floor(Nx/L*L_1)
	Ni_z = math.ceil((1-H_2/H)*Nz)
	Nf_z = math.floor((1-H_1/H)*Nz)
	return Nf_x, Ni_x, Nf_z, Ni_z
def create_domain(L_1, L_2, H_1, H_2):
	Nf_x, Ni_x, Nf_z, Ni_z = generate_modes(L_1, L_2, H_1, H_2)
	xbasis = de.Fourier('x', Nf_x-Ni_x, interval=(L_1, L_2))
	zbasis = de.Fourier('z', Nf_z-Ni_z, interval=(H_1, H_2))
	return de.Domain([xbasis, zbasis], grid_dtype=np.float64)

def create_data_file():
	h5_files = sort_h5files(files, splice0, splice1)
	file = open('potentialdata_{0}_{1}-{2}.txt'.format(fileno, splice0, splice1), 'a+')
	data = []
	for _ in range(25):
		data.append([])
	for h5file in h5_files[::3]:
		print(h5file)
		with h5py.File(h5file, mode='r') as f: 
			times = f['tasks']['u'].dims[0]['sim_time'][:]
			#Region 1: Total domain
			L_1, L_2, H_1, H_2 = 10, L-10, 0, H
			energies = compute_potential_energies(h5file, L_1, L_2, H_1, H_2)
			ML = compute_mixedlayerdepth(h5file, L_1, L_2, H_1, H_2)
			data[0].append(times[0])
			data[1].append(energies['E_b'])
			data[2].append(energies['E_a'])
			data[3].append(energies['E_p'])
			data[4].append(energies['phi_d'])
			data[5].append(energies['phi_z'])
			data[6].append(energies['ked'])
			data[7].append(ML[0])
			data[8].append(ML[1])
			data[9].append(energies['K'])
			data[10].append(energies['E_k'])
			#Region 2: Downstream from keel height
			
			L_1, L_2, H_1 = l, L-40, 0
			Zm = json.load(open('Zm_values_{0}-{1}.txt'.format(L_1, L_2)))
			H_2 = Zm[a_s+c_s]
			print(a_s+c_s, H_2/(H-z0))
			energies = compute_potential_energies(h5file, L_1, L_2, H_1, H_2)
			data[11].append(energies['E_b'])
			data[12].append(energies['E_a'])
			data[13].append(energies['E_p'])
			data[14].append(energies['phi_d'])
			data[15].append(energies['phi_z'])
			data[16].append(energies['ked'])
			data[17].append(energies['K'])
			#Region 3: Upstream from keel height
			L_1, L_2, H_1 = 160, l, 0
			energies = compute_potential_energies(h5file, L_1, L_2, H_1, H_2)
			data[18].append(energies['E_b'])
			data[19].append(energies['E_a'])
			data[20].append(energies['E_p'])
			data[21].append(energies['phi_d'])
			data[22].append(energies['phi_z'])
			data[23].append(energies['ked'])
			data[24].append(energies['K'])
	MLD_std = np.std(data[8])
	print('Writing to file')
	file.write(str(MLD_std)+'\nTime \t Eb \t Ea \t Ep \t phi_d \t phi_z \t KED \t MLD \t K \t E_k \t Eb_D \t Ea_D \t Ep_D \t phi_d_D \t phi_z_D \t KED_D \t K_D \t Eb_U \t Ea_U \t Ep_U \t phi_d_U \t phi_z_U \t KED_U \t K_U \n')
	for i in range(len(data[0])):
		file.write(str(data[0][i])+'\t'+str(data[1][i])+'\t'+str(data[2][i])+'\t'+str(data[3][i])+'\t'+str(data[4][i])+
		'\t'+str(data[5][i])+'\t'+str(data[6][i])+'\t' +str(data[7][i])+'\t'+str(data[9][i])+'\t'+str(data[10][i])+'\t'+
		str(data[11][i])+'\t'+str(data[12][i])+'\t'+str(data[13][i])+'\t'+str(data[14][i])+'\t'+str(data[15][i])+
		'\t'+str(data[16][i])+'\t'+str(data[17][i])+'\t'+str(data[18][i])+'\t'+str(data[19][i])+'\t'+str(data[20][i])+
		'\t'+str(data[21][i])+'\t'+str(data[22][i])+'\t'+str(data[23][i])+'\t'+str(data[24][i])+'\n')
	file.close()
	print('Done')

def create_salt_file():
	h5_files = sort_h5files(files, splice0, splice1)
	zdata = []
	time = []
	for fil in h5_files[::3]:
		with h5py.File(fil, mode='r') as f:
			print(fil)
			zdata.append(compute_mixedlayerdepth(fil, 60*8, 70*8, 0, 80)[0])
			time.append(f['tasks']['u'].dims[0]['sim_time'][:][0])
	plt.plot(np.array(time)/t_0, np.array(zdata)/(H-z0))
	plt.xlabel('$t/t_0$')
	plt.ylabel('Mixed Layer/$z_0$')
	plt.savefig('ML.png')
	plt.clf()

def mixing_depth_calculate(rho, L_1, L_2, ab):
	phi_d = []
	phi_d_avg = []
	for z_m in np.linspace(0, H, Nz)[3:-100:3]:
		Nf_x, Ni_x, Nf_z, Ni_z = generate_modes(L_1, L_2, 0, z_m)
		domain = create_domain(L_1, L_2, 0, z_m)
		x, z = domain.grids(domain.dealias)
		integrand = domain.new_field()
		integrand.set_scales(domain.dealias)
		h = (H-z0)*conv(ab)
		region = 0.5*(1-np.tanh((z-H+h*gen_sigma(ab)**2/(gen_sigma(ab)**2+4*(x-l)**2))/0.01))
		integrand['g'] = region
		A = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]
		rho_ref, z_ref = sort_rho_z(rho, L_1, L_2, 0, z_m)
		rho_cut = rho[Ni_x:Nf_x, Ni_z:Nf_z]
		deriv = 1/np.gradient(rho_ref, z[0], axis=1)
		rho_z = np.gradient(rho_cut, z[0], axis=1)
		rho_x = np.gradient(rho_cut, np.concatenate(x).ravel(), axis=0)
		integrand['g'] = -9.8*mu*np.nan_to_num(deriv, nan=0, posinf=0, neginf=0)*(rho_x**2+rho_z**2)*region/(sw.dens0(28,-2))
		phi_dval = de.operators.integrate(integrand, 'x', 'z').evaluate()['g'][0][0]
		#rho_avg_s = np.gradient(np.average(rho_ref, axis=0), z[0])
		#N_sq = -9.8/sw.dens0(28,-2)*np.average(rho_avg_s)
		phi_d.append(phi_dval)
		phi_d_avg.append(phi_dval/A)
	return (np.array(phi_d), np.array(phi_d_avg))

def create_phi_d_json(L_1, L_2):
	a_s = ['005', '095', '102', '200']
	c_s = ['005', '100', '105', '200']
	times = [[220, 260, 450, 450], [220, 260, 450, 450], [220, 260, 450, 450], [220, 260, 450, 450]]
	phi_d = {}
	phi_d_avg = {}
	for i in range(len(a_s)):
		for j in range(len(c_s)):
			for k in range(70, times[i][j], 10):
				l = 0
				phi_d_temp = 0
				phi_d_temp_avg = 0
				with h5py.File('new/data-mixingsim-a{0}c{1}-00/data-mixingsim-a{0}c{1}-00_s{2}.h5'.format(a_s[i], c_s[j], k), mode='r') as f:
					phi_d_temp += mixing_depth_calculate(f['tasks']['rho'][0], L_1, L_2, a_s[i])[0]
					phi_d_temp_avg += mixing_depth_calculate(f['tasks']['rho'][0], L_1, L_2, a_s[i])[1]		
					l += 1
			phi_d[a_s[i]+c_s[j]] = phi_d_temp/l
			phi_d_avg[a_s[i]+c_s[j]] = phi_d_temp_avg/l
			print(i,j)
	for key in phi_d.keys():
            phi_d[key] = list(phi_d[key])
            phi_d_avg[key] = list(phi_d_avg[key])
	json.dump(phi_d, open('phi_d_values_{0}-{1}.txt'.format(L_1, L_2), 'w'))
	json.dump(phi_d_avg, open('phi_d_avg_values_{0}-{1}.txt'.format(L_1, L_2), 'w'))

def create_Zm_json(L_1, L_2):
	phi_d_avg = json.load(open('phi_d_avg_values_{0}-{1}.txt'.format(L_1, L_2)))
	a_s = ['005', '095', '102', '200']
	c_s = ['005', '100', '105', '200']
	Zm = {}
	for ab in a_s:
		for cb in c_s:
			peaks, _ = sc.find_peaks(np.array(phi_d_avg[ab+cb]), height=np.average(phi_d_avg[ab+cb]), width=(3, 1e6))
			Zm[ab+cb] = int(np.linspace(0, H, Nz)[3:-100:3][peaks[-1]])	
			print(ab+cb, peaks, phi_d_avg[ab+cb][peaks[-1]], Zm[ab+cb]/(H-z0))
			#v = np.argwhere(np.array(phi_d_avg[ab+cb]) >= 1)
			#Zm[ab+cb] = int(np.min(v))
	json.dump(Zm, open('Zm_values_{0}-{1}.txt'.format(L_1, L_2), 'w'))

def generate_new_set():
	print('Building upstream phi_d json')
	#create_phi_d_json(160, l)
	print('Building upstream Zm json')
	create_Zm_json(160, l)
	print('Building downstream phi_d json')
	#create_phi_d_json(l, L-40)
	print('Building downstream Zm json')
	create_Zm_json(l, L-40)

def mixing_depth_figure(L_1, L_2, title):
	plt.rcParams.update({'font.size': 12})
	a_s = ['005', '095', '102', '200']
	c_s = ['005', '100', '105', '200']
	#colors = {"200": "#FF1300", "102": "#0CCAFF", "095": "#29E91F", "005": "#a67acf"}
	colors = {'005': '#99c0ff', '095': '#3385ff', '102': '#0047b3', '200': '#000a1a'}
	styles = {"200": "solid", "105": (0, (1,1)), "100": "dashed", "005": (0, (3, 1, 1, 1))}
	z = np.linspace(0, H, Nz)
	phi_d = json.load(open('phi_d_avg_values_{0}-{1}.txt'.format(L_1, L_2)))
	Zm = json.load(open('Zm_values_{0}-{1}.txt'.format(L_1, L_2)))
	fig, ax1 = plt.subplots()
	for ab in a_s[::-1]:
		for cb in c_s:
			ax1.plot(np.array(phi_d[ab+cb]), (z[3:-100:3])/(H-z0), color=colors[ab], linestyle=styles[cb], linewidth=2)
	ax1.set_xscale('log')
	ax2 = ax1.twiny()
	x = np.linspace(0, 120, Nx)
	ax2.fill_between(x, 0, -0.01+2*gen_sigma(2)**2/(gen_sigma(2)**2+(H-z0)**2*(x-115)**2), facecolor='w')
	ax2.fill_between(x, 0, -0.01+2*gen_sigma(2)**2/(gen_sigma(2)**2+(H-z0)**2*(x-115)**2), facecolor=colors['200'], alpha=0.9)
	ax2.fill_between(x, 0, -0.01+1.2*gen_sigma(1.2)**2/(gen_sigma(1.2)**2+(H-z0)**2*(x-115)**2), facecolor='w')
	ax2.fill_between(x, 0, -0.01+1.2*gen_sigma(1.2)**2/(gen_sigma(1.2)**2+(H-z0)**2*(x-115)**2), facecolor=colors['102'], alpha=0.9)
	ax2.fill_between(x, 0, -0.01+0.95*gen_sigma(0.95)**2/(gen_sigma(0.95)**2+(H-z0)**2*(x-115)**2), facecolor='w')
	ax2.fill_between(x, 0, -0.01+0.95*gen_sigma(0.95)**2/(gen_sigma(0.95)**2+(H-z0)**2*(x-115)**2), facecolor=colors['095'], alpha=0.9)
	ax2.fill_between(x, 0, -0.01+0.5*gen_sigma(0.5)**2/(gen_sigma(0.5)**2+(H-z0)**2*(x-115)**2), facecolor='w')
	ax2.fill_between(x, 0, -0.01+0.5*gen_sigma(0.5)**2/(gen_sigma(0.5)**2+(H-z0)**2*(x-115)**2), facecolor=colors['005'], alpha=0.9)
	#plt.plot(x, -np.ones(len(x)), linestyle='dashed', color='black')
	ax2.plot([], [], linestyle=styles['200'], label="$Fr=2$", color='black')
	ax2.plot([], [], linestyle=styles['105'], label="$Fr=1.5$", color='black')
	ax2.plot([], [], linestyle=styles['100'], label="$Fr=1$", color='black')
	ax2.plot([], [], linestyle=styles['005'], label="$Fr=0.5$", color='black')
	ax2.set_xticks([])
	ax2.set_xlim(0, 120)
	plt.legend(loc='lower right', fancybox=True, shadow=True, prop={'size':11})
	ax1.set_xlim(1e-6, 1e-1)
	ax1.set_ylim(0, 7)
	ax1.set_ylim(plt.ylim()[::-1])
	ax1.set_xlabel(title+" Average Mixing Rate $\\Phi_d$ (J s$^{{-1}}$)".format(title[0]))
	ax1.set_title('(a)')
	ax1.set_ylabel("Vertical Depth $z/z_0$")
	ax1.grid(visible=True)
	#ax1.set_title(title)
	fig.set_size_inches(6.4,4.8)
	plt.savefig('Kp_depth_figure_{0}-{1}.png'.format(L_1,L_2), dpi=600, bbox_inches='tight')
	plt.clf()
	#Max mixing depth figure
	c_v = {'005': 0.5, '100': 1.0, '105': 1.5, '200': 2.0}
	if L_1 == 160:
		markers = {'005005': 'D', '005100': 'D', '005105': 'D', '005200': 'P', '095005': 'D', '095100': 'o', '095105': 'o', '095200': 'P', '102005': 'D', '102100': 'o', '102105': 'o', '102200': 'P', '200005': 's', '200100': 'o', '200105': 'o', '200200': 'P'}
	else:
		markers = {'005005': 'd', '005100': '^', '005105': '^', '005200': '*', '095005': 'd', '095100': '^', '095105': 'p', '095200': '*', '102005': 'd', '102100': '^', '102105': 'p', '102200': '*', '200005': 's', '200100': '^', '200105': 'p', '200200': '*'}
	colors2 = ['#99c0ff', '#3385ff', '#0047b3', '#000a1a']
	labels = ['$\\eta=0.5$', '$\\eta=0.95$', '$\\eta=1.2$', '$\\eta=2.0$']
	i = 0 
	for ab in a_s:
		for cb in c_s:
			print(ab+cb, (H-Zm[ab+cb]/Nz*H)/((H-z0)*conv(ab)))
			plt.plot(c_v[cb], Zm[ab+cb]/(H-z0), marker=markers[ab+cb], color=colors2[i], ms=7)
		i += 1
	for i in range(len(a_s)):
		plt.plot([], [], marker='X', linestyle='None', color=colors2[i], label=labels[i])
	if L_1 != 160:
		plt.plot([], [], marker='o', label=' ', color='white')
		for i in range(5):
			plt.plot([], [], marker=['*', 'p', '^', 's', 'd'][i], linestyle='None', color='black', label=['Vortex Shedding', 'Stirred', 'Laminar Jump', 'Blocked', 'Lee Waves'][i])
	else:
		for i in range(4):
			plt.plot([], [], marker=['P', 'D', 'o', 's'][i], linestyle='None', color='black', label=['Supercritical', 'Rarefaction', 'Solitary Waves', 'Blocked'][i])
	plt.xlabel('Froude Number $Fr$')
	plt.ylabel('Maximum Mixing Depth $z_{{max}}/z_0$')
	plt.grid()
	plt.title('(b)')
	plt.ylim(plt.ylim()[0], 6)
	handles, labels_temp = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels_temp, handles))
	plt.legend(by_label.values(), by_label.keys(), fancybox=True, shadow=True, prop={'size': 9}, loc='upper left', ncol=2, handleheight=0.5)
	plt.xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
	#if L_1 == 160:
		#plt.title('Upstream')
	#else:
		#plt.title('Downstream')
	plt.savefig('Max_mixing_depth_{0}-{1}.png'.format(L_1,L_2), dpi=600, bbox_inches='tight')
	plt.clf()
 
	
		 




create_data_file()
#generate_new_set()
#mixing_depth_figure(160, l, 'Upstream')
#mixing_depth_figure(l, L-40, 'Downstream')
#create_salt_file()
#plot_potential_energies()
