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
import scipy.signal as sc
import seawater as sw
import json
from scipy.interpolate import UnivariateSpline

#################################
# Constants

z0 = 8 # Depth of mixed layer [m]
rho1 = 1022.49 # Density of mixed layer (used as reference density rho0)
rho2 = 1024.12 # Density of deeper ocean
mu = 2e-3 # Salt-mass diffusivity [m^2/s]
g = 9.8 # Acceleration due to gravity [m/s^2]
L = 960 # Domain length [m]
H = 80 # Domain height [m]
l = 75*z0 # Keel center location [m]
Nx = 1280 # Number of grid points in horizontal
Nz = 640 # Number of grid points in vertical
Nx_f = math.ceil(Nx/L*(L-5*z0))
Nx_i = math.floor(Nx/L*(20*z0))
x = np.linspace(0, L, Nx) # Horizontal grid points [m]
z = np.linspace(0, H, Nz) # Vertical grid points (increasing downwards) [m]

def keel(h, l, x):  # Eqn for keel (from SD)
    """
    h: Keel maximum height
    l: Where the keel is centered in the domain
    """
    sigma = 3.9 * h # Keel characteristic width
    return h*sigma**2/(sigma**2+4*(x-l)**2)

def find_mask(h, l, Nx, Nz, zv): # Mask out the keel based on cell height
    keel_height = keel(h, l, x)
    for ind in range(Nx):
        for indz in range(Nz):
            if zv[ind,indz]<=keel_height[ind]:
                zv[ind,indz]=np.NaN
    zv_masked = np.ma.masked_invalid(zv)
    keel_mask = np.ma.getmask(zv_masked)
    return keel_mask  # returns mask of which elements are within the keel

dx = L/Nx # Grid spacing in x
dz = H/Nz # Grid spacing in z
vol = dx*dz # "Volume" (area) of each grid cell

zv, xv = np.meshgrid(z,x) # create meshgrid of (x,z) coordinates

Nx_mid = int(np.where(np.abs(x-l) == np.min(np.abs(x-l)))[0])

#################################
# Functions

def conv(a_s):
	# Converts string names to numerical values
	if a_s == "a200":
		return 2
	elif a_s == "a105":
		return 1.5
	elif a_s == "a102":
		return 1.2
	elif a_s == "a100":
		return 1.0
	elif a_s == "a095":
		return 0.95
	elif a_s == "a005":
		return 0.5


def pad(data):
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data[bad_indexes] = interpolated
    return data

def find_zstar(rho,h,l,Nx,Nz,zv,Nx1,Nx2,dz,vol,upstream_flag,keel_mask): 
    # Find the profile z*(rho) and dz*/drho for each non-masked element
    #       within the domain [Nx1:Nx2, :] (can separate down- and upstream)
    
    # upstream_flag = 1 (= 0) looking at upstream (downstream)
    #   Need this flag to distinguish positive and negative roots of the keel
    #       function
    
    rho_masked = np.ma.array(rho,mask=keel_mask) #mask the keel in rho
    sigma = 3.9*h
    num_elem = (Nx2-Nx1)*Nz #number of elements in this domain
    
    #get rho in the proper domain part (upstream or downstream), reshape
    #   as a column vector, and fill the masked parts with NaN
    rho_dom_col = rho_masked[Nx1:Nx2,:].reshape((num_elem,1)).filled(np.NaN)
    
    #column vector of elements numbers, so that we can keep track of where
    #   each cell goes back to in the physical (x,z) space
    elem_ind = np.arange(num_elem).reshape((num_elem,1))
    
    #remove the elements that are within the keel (marked with NaNs)
    elem_ind = elem_ind[~np.isnan(rho_dom_col)]
    rho_dom_col = rho_dom_col[~np.isnan(rho_dom_col)]
    
    n = len(rho_dom_col) # How many elements we have not in the keel
    #column vector of each cell volume (here all cells are the same)
    vol_ar = vol*np.ones((n,1))
    #put all three columns (rho, cell volume, and cell element number) together
    ar = np.concatenate((rho_dom_col.reshape(n,1),
                         vol_ar,
                         elem_ind.reshape(n,1)),
                        axis=1)
    
    #sort in the DESCENDING order of density \rho
    ind_rho = np.argsort(-ar[:,0] )
    ar = ar[ind_rho]
    
    #Find the "area" (horizontal distance in 2D case) unoccupied by the keel
    #   at each vertical level
    Area = np.zeros((Nz,))
    for i in range(1,Nz):
        if z[i]<=h:
            # x(z) for the keel -> inverse of keel function given above
            if upstream_flag==1:
                Area[i] = -0.25*np.sqrt((h*sigma**2)/z[i] - sigma**2) + l
            else:
                Area[i] = x[Nx2] - (0.25*np.sqrt((h*sigma**2)/z[i] - sigma**2)+l)
        else:
            Area[i] = np.abs(x[Nx2] - x[Nx1])    
    Area = np.flip(Area)
    Volz = Area*dz
    
    #Now let the stacking of the cells begin!
    totvol = 0.0  #"volume" (area in 2D) tracker
    ind_a = 0 #vertical level tracker
    #vector for each cell -> to be filled with \delta z (vertical space) each 
    #   cell will occupy
    dzs = np.zeros((n,))
    
    for i in range(n):
        totvol += ar[i,1]
        if totvol < Volz[ind_a]:
            dzs[i] = ar[i,1]/Area[ind_a]
        else:
            if ind_a < Nz-1:
                ind_a +=1
                dzs[i] = ar[i,1]/Area[ind_a]
                totvol = 0.0
            else:
                dzs[i] = ar[i,1]/Area[ind_a]
     
    #Now we have for each element, stacked from the densest to the lightest,
    #   the vertical spacing (\delta z) that it occupies.
    
    #z* is just a cumulative sum of these vertical spacings, i.e., how far
    #   above the bottom is a particular element stacked
    z_star = -(np.cumsum(dzs)-H) #sign change etc needed because of the particular
            #coordinate system here
            
    #applying some spline interpolation so that the derivative dz*/drho is smoother
    spl2 = UnivariateSpline(np.flip(ar[:,0]),np.flip(z_star),k=5,s=1.8)
    dzsdb = spl2.derivative(1)(ar[:,0]) #this is dz*/drho
    #adjust for minor discrepancy of negative gradients 
    #   introduced by spline interpolation
    dzsdb[dzsdb<0] = np.NaN
    dzsdb = np.apply_along_axis(pad, 0, dzsdb)

    #Now put together in one array three column vectors:
    #       (z*, element number, dz*/drho) 
    ZS_tot = np.concatenate((np.reshape(z_star,(n,1)),
                         np.reshape(ar[:,2],(n,1)),
                         np.reshape(dzsdb,(n,1))),
                    axis=1)

    
    #Using the element number, find where dz*/drho belongs in physical (x,z) space.
    #This is the needed value dz*/drho evaluated at each local rho
    dzdb_final_col = np.nan*np.ones((num_elem,))
    dzdb_final_col[ZS_tot[:,1].astype(int)] = ZS_tot[:,2]
    dzdb_final= dzdb_final_col.reshape((Nx2-Nx1,Nz))
  
    return ar[:,0], z_star, dzdb_final #return column vectors of \rho and z*
                # and dz*/drho(x,z) matrix

def rho_deriv(rho,x,z):
    rho_z = np.gradient(rho,z,axis=1);
    rho_x = np.gradient(rho,x,axis=0);
    nabla_rho = (rho_z)**2 + (rho_x)**2
    return nabla_rho

def calc_mixing(rho,h,l,Nx,Nz,zv,Nx1,Nx2,dz,vol,upstream_flag,x,z,keel_mask):
    b, zs, dzdb = find_zstar(rho,h,l,Nx,Nz,zv,Nx1,Nx2,dz,vol,upstream_flag,keel_mask)
    nabla_rho = rho_deriv(rho,x,z)

    mixing = nabla_rho[Nx1:Nx2,:]*dzdb
    
    return b, zs,  mixing

def find_rhostar(rho, Nx1, Nx2, keel_mask):
    rho_masked = np.transpose(np.ma.array(rho, mask=keel_mask)[Nx1:Nx2])
    rho_sorted = np.sort(rho_masked[rho_masked.mask == False])
    rho_star = (0.0*rho_masked).filled(np.nan)
    inds = np.argwhere(rho_star==0.0)
    rho_star[tuple(np.transpose(np.array(inds)))] = rho_sorted
    rho_star = np.transpose(rho_star)
    rho_star = np.ma.array(rho_star, mask=np.isnan(rho_star))
    return np.ma.mean(rho_star, axis=0)

def mixing_format(rho, ab):
    rho = np.array(rho)[:, ::-1]
    h = conv(ab)*z0
    keel_mask = find_mask(h, l, Nx, Nz, zv)
    
    #Calculate upstream
    b_up, zs_up, mixing_up = calc_mixing(rho, h, l, Nx, Nz, zv, Nx_i, Nx_mid, dz, vol, 1, x, z, keel_mask)

	#Calculate downstream
    b_down, zs_down, mixing_down = calc_mixing(rho,h,l,Nx,Nz,zv,Nx_mid,Nx_f,dz,vol,0,x,z, keel_mask)

    mixing_up_ma = np.ma.array(mixing_up,mask=keel_mask[Nx_i:Nx_mid,:])
    mixing_dn_ma = np.ma.array(mixing_down,mask=keel_mask[Nx_mid:Nx_f,:])

    tot_mix_up = np.sum(mixing_up_ma)*vol*g*mu/(rho1*mixing_up_ma.size) #tot mixing upstream
    tot_mix_dn = np.sum(mixing_dn_ma)*vol*g*mu/(rho1*mixing_dn_ma.size) #tot mixing downstream

    rho_star_up_z = np.gradient(find_rhostar(rho, Nx_i, Nx_mid, keel_mask), z)
    rho_star_dn_z = np.gradient(find_rhostar(rho, Nx_mid, Nx_f, keel_mask), z)
    
    N_star_sq_up = g/rho1 * np.ma.average(rho_star_up_z) 
    N_star_sq_down = g/rho1 * np.ma.average(rho_star_dn_z) 
    
    tot_diff_up = tot_mix_up/(N_star_sq_up*mu)
    tot_diff_dn = tot_mix_dn/(N_star_sq_down*mu)

    #z_mix
    ind_up = np.argwhere(np.cumsum(np.sum(mixing_up_ma, axis=0)) > 0.95*np.sum(mixing_up_ma))[0][0]
    z_mix_up = z[ind_up]
    ind_dn = np.argwhere(np.cumsum(np.sum(mixing_dn_ma, axis=0)) > 0.95*np.sum(mixing_dn_ma))[0][0]
    z_mix_dn = z[ind_dn]
   
    return (float(tot_mix_up), float(tot_mix_dn), float(tot_diff_up), float(tot_diff_dn), float(z_mix_up), float(z_mix_dn))

def create_jsons():
    a_s = ['a005', 'a095', 'a102', 'a200']
    c_s = ['c005', 'c100', 'c105', 'c200']
    times = [220, 260, 450, 450] # Run times in file count
    K_up = {}
    K_dn = {}
    z_mix_up = {}
    z_mix_dn = {}
    phi_d_up = {}
    phi_d_dn = {}
    for i in range(len(a_s)):
        for j in range(len(c_s)):
            K_up_temp = []
            K_dn_temp = []
            z_mix_up_temp = []
            z_mix_dn_temp = []
            phi_d_up_temp = []
            phi_d_dn_temp = []
            for k in range(70, times[j], 3): # Loop through all files
                with h5py.File('new/data-mixingsim-{0}{1}-00/data-mixingsim-{0}{1}-00_s{2}.h5'.format(a_s[i], c_s[j], k), mode='r') as f:
                    temp = mixing_format(f['tasks']['rho'][0], a_s[i])
                    phi_d_up_temp.append(temp[0])
                    phi_d_dn_temp.append(temp[1])
                    K_up_temp.append(temp[2])
                    K_dn_temp.append(temp[3])
                    z_mix_up_temp.append(temp[4])
                    z_mix_dn_temp.append(temp[5])
		    # Compute average quantities
            K_up[a_s[i]+c_s[j]] = K_up_temp
            K_dn[a_s[i]+c_s[j]] = K_dn_temp
            z_mix_up[a_s[i]+c_s[j]] = z_mix_up_temp
            z_mix_dn[a_s[i]+c_s[j]] = z_mix_dn_temp
            phi_d_up[a_s[i]+c_s[j]] = phi_d_up_temp
            phi_d_dn[a_s[i]+c_s[j]] = phi_d_dn_temp
            print(i,j)
	# Store data in json formatted as dictionaries 
    json.dump(K_up, open('K_values_{0}-{1}.txt'.format(160, 600), 'w'))
    json.dump(K_dn, open('K_values_{0}-{1}.txt'.format(600, 920), 'w'))
    json.dump(z_mix_up, open('z_mix_values_{0}-{1}.txt'.format(160, 600), 'w'))
    json.dump(z_mix_dn, open('z_mix_values_{0}-{1}.txt'.format(600, 920), 'w'))
    json.dump(phi_d_up, open('phi_d_values_{0}-{1}.txt'.format(160, 600), 'w'))
    json.dump(phi_d_dn, open('phi_d_values_{0}-{1}.txt'.format(600, 920), 'w'))

def generate_new_set():
	# Generates diapycnal diffusivitiy and zmix json for upstream and downstream
    print('Building json')
    create_jsons()
	

# Run
if __name__ == "__main__":
	#generate_new_set()
	#rho = json.load(open('F20H20_rho_220.txt')) # F20H20 (vortex shedding regime) at snapshot t=220t_0 (1280x640 2D array)
	#rho = np.array(rho)[:, ::-1] # Reorder so rho[0][0] is top left of domain
    #with h5py.File('regime_files/data-mixingsim-a005c005-00_s180.h5', mode='r') as f:
    #    rho = f['tasks']['rho'][0]
    #print(mixing_format(rho, 'a005'))
    generate_new_set()
