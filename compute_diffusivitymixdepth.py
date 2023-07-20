"""
Computes diapycnal mixing rates, diffusivity values, and mixing depth for all 16 simulations.

Authors: Barbara Zemskova & Sam De Abreu (2023)

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
"""
#################################
# Imports
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
DB = g*(rho2-rho1)/rho1
t0 = np.sqrt(z0/DB)
Nx = 1280 # Number of grid points in horizontal
Nz = 640 # Number of grid points in vertical
Nx_f = math.ceil(Nx/L*(L-5*z0))
Nx_i = math.floor(Nx/L*(20*z0))
x = np.linspace(0, L, Nx) # Horizontal grid points [m]
z = np.linspace(0, H, Nz) # Vertical grid points (increasing downwards) [m]

dx = L/Nx # Grid spacing in x
dz = H/Nz # Grid spacing in z
vol = dx*dz # "Volume" (area) of each grid cell

zv, xv = np.meshgrid(z,x) # create meshgrid of (x,z) coordinates

Nx_mid = int(np.where(np.abs(x-l) == np.min(np.abs(x-l)))[0])

conv_id = {'a005': 'H05', 'a095': 'H09', 'a102': 'H12', 'a200': 'H20', 'c005': 'F05', 'c100': 'F10', 'c105': 'F15', 'c200': 'F20'}

#################################
# Functions

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
    
def name_to_h(name,z0):
    if name[1::]=='005':
        h = 0.5*z0
    elif name[1::]=='095':
        h = 0.95*z0
    elif name[1::]=='102':
        h = 1.2*z0
    else:
        h = 2.0*z0
    return h

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

def reject_outliers(data, m=2):
    data = np.ma.masked_invalid(data)
    data = np.ma.masked_where(abs(data - np.ma.mean(data)) > m * np.ma.std(data),data)
    return data #[abs(data - np.ma.mean(data)) < m * np.ma.std(data)]

def reject_outliers2(data, m=2):
    data = np.ma.masked_invalid(data)
    data_sort = np.ma.sort(data.reshape((data.size,))).filled(np.nan)
    data_sort = data_sort[~np.isnan(data_sort)]
    cutoff = data_sort[-15]
    data = np.ma.masked_where(data>=cutoff, data)
    #np.ma.masked_where(abs(data - np.ma.mean(data)) > m * np.ma.std(data),data)
    return data
    
def calc_mixing(rho,h,l,Nx,Nz,zv,Nx1,Nx2,dz,vol,upstream_flag,x,z,keel_mask):
    b, zs, dzdb = find_zstar(rho,h,l,Nx,Nz,zv,Nx1,Nx2,dz,vol,upstream_flag,keel_mask)
    nabla_rho = rho_deriv(rho,x,z)
    
    #Try removing outliers one of two ways:
    #Way 1: directly find outliers
    #dzdb_masked = np.ma.array(dzdb,mask=keel_mask[Nx1:Nx2,:])
    #dzdb_rem_outliers = reject_outliers(dzdb_masked)
    
    #Way 2: remove the largest 10 points
    dzdb_masked = np.ma.array(dzdb,mask=keel_mask[Nx1:Nx2,:])
    
    mixing = nabla_rho[Nx1:Nx2,:]*dzdb_masked
    mixing_masked = np.ma.array(mixing,mask=keel_mask[Nx1:Nx2,:])
    mixing = reject_outliers2(mixing_masked)
    
    return b, zs,  mixing, dzdb

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
    h = name_to_h(ab,z0) 
    keel_mask = find_mask(h, l, Nx, Nz, zv)
    
    #Calculate upstream
    b_up, zs_up, mixing_up, dzdb_up = calc_mixing(rho, h, l, Nx, Nz, zv, Nx_i, Nx_mid, dz, vol, 1, x, z, keel_mask)

	#Calculate downstream
    b_down, zs_down, mixing_down, dzdb_down = calc_mixing(rho,h,l,Nx,Nz,zv,Nx_mid,Nx_f,dz,vol,0,x,z, keel_mask)

    mixing_up_ma = np.ma.array(mixing_up,mask=keel_mask[Nx_i:Nx_mid,:])
    mixing_dn_ma = np.ma.array(mixing_down,mask=keel_mask[Nx_mid:Nx_f,:])
    
    #Now area-averaged in the right units
    tot_mix_up = np.sum(mixing_up_ma)*g*mu/(rho1*mixing_up_ma.size) #tot mixing upstream
    tot_mix_dn = np.sum(mixing_dn_ma)*g*mu/(rho1*mixing_dn_ma.size) #tot mixing downstream

    
    dbdz_up = 1/dzdb_up
    dbdz_dn = 1/dzdb_down
    
    dbdz_up = np.ma.array(dbdz_up,mask=keel_mask[Nx_i:Nx_mid,:])
    dbdz_up = reject_outliers2(dbdz_up)
    
    dbdz_dn = np.ma.array(dbdz_dn,mask=keel_mask[Nx_mid:Nx_f,:])
    dbdz_dn = reject_outliers2(dbdz_dn)
    
    N_star_sq_up = (g/rho1)*(np.ma.mean(dbdz_up))
    N_star_sq_down = (g/rho1)*(np.ma.mean(dbdz_dn))
    
    tot_diff_up = tot_mix_up/(N_star_sq_up*mu)
    tot_diff_dn = tot_mix_dn/(N_star_sq_down*mu)

    #z_mix
    ind_up = np.argwhere(np.cumsum(np.ma.sum(mixing_up_ma, axis=0)) > 0.95*np.ma.sum(mixing_up_ma))[0][0]
    z_mix_up = z[ind_up]
    ind_dn = np.argwhere(np.cumsum(np.sum(mixing_dn_ma, axis=0)) > 0.95*np.sum(mixing_dn_ma))[0][0]
    z_mix_dn = z[ind_dn]

    #z_mix_rel N^2 version
    N_sq = np.gradient(rho, axis=1)
    #Upstream
    inds_max_up = np.argmax(N_sq[Nx_i:Nx_mid], axis=1)
    z_mix_rel_up = np.max(z[inds_max_up] - keel(h, l, x[Nx_i:Nx_mid]))
    x_mix_rel_up = x[Nx_i+np.argmax(z[inds_max_up] - keel(h, l, x[Nx_i:Nx_mid]))]
    #Downstream
    inds_max_dn= np.argmax(N_sq[Nx_mid:Nx_f], axis=1)
    z_mix_rel_dn = np.max(z[inds_max_dn] - keel(h, l, x[Nx_mid:Nx_f]))
    x_mix_rel_dn = x[Nx_mid+np.argmax(z[inds_max_dn] - keel(h, l, x[Nx_mid:Nx_f]))]
    
    return float(tot_mix_up), float(tot_mix_dn), float(tot_diff_up), float(tot_diff_dn), float(z_mix_up), float(z_mix_dn), float(N_star_sq_up), float(N_star_sq_down), float(z_mix_rel_up), float(z_mix_rel_dn)
            

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
    Nstar_sq_up = {}
    Nstar_sq_dn = {}
    z_mix_rel_up = {}
    z_mix_rel_dn = {}
    for i in range(len(a_s)):
        for j in range(len(c_s)):
            K_up_temp = []
            K_dn_temp = []
            z_mix_up_temp = []
            z_mix_dn_temp = []
            phi_d_up_temp = []
            phi_d_dn_temp = []
            Nstar_sq_up_temp = []
            Nstar_sq_dn_temp = []
            z_mix_rel_up_temp = []
            z_mix_rel_dn_temp = []
            time = []
            for k in range(70, times[j], 3): # Loop through all files
                with h5py.File('new/data-mixingsim-{0}{1}-00/data-mixingsim-{0}{1}-00_s{2}.h5'.format(a_s[i], c_s[j], k), mode='r') as f:
                    temp = mixing_format(f['tasks']['rho'][0], a_s[i])
                    time.append(f['tasks']['rho'].dims[0]['sim_time'][0]/t0)
                    phi_d_up_temp.append(temp[0])
                    phi_d_dn_temp.append(temp[1])
                    K_up_temp.append(temp[2])
                    K_dn_temp.append(temp[3])
                    z_mix_up_temp.append(temp[4])
                    z_mix_dn_temp.append(temp[5])
                    Nstar_sq_up_temp.append(temp[6])
                    Nstar_sq_dn_temp.append(temp[7])
                    z_mix_rel_up_temp.append(temp[8])
                    z_mix_rel_dn_temp.append(temp[9])
		    # Compute average quantities
            K_up[conv_id[c_s[j]]+conv_id[a_s[i]]] = (K_up_temp, time)
            K_dn[conv_id[c_s[j]]+conv_id[a_s[i]]] = (K_dn_temp, time)
            z_mix_up[conv_id[c_s[j]]+conv_id[a_s[i]]] = (z_mix_up_temp, time)
            z_mix_dn[conv_id[c_s[j]]+conv_id[a_s[i]]] = (z_mix_dn_temp, time)
            phi_d_up[conv_id[c_s[j]]+conv_id[a_s[i]]] = (phi_d_up_temp, time)
            phi_d_dn[conv_id[c_s[j]]+conv_id[a_s[i]]] = (phi_d_dn_temp, time)
            Nstar_sq_up[conv_id[c_s[j]]+conv_id[a_s[i]]] = (Nstar_sq_up_temp, time)
            Nstar_sq_dn[conv_id[c_s[j]]+conv_id[a_s[i]]] = (Nstar_sq_dn_temp, time)
            z_mix_rel_up[conv_id[c_s[j]]+conv_id[a_s[i]]] = (z_mix_rel_up_temp, time)
            z_mix_rel_dn[conv_id[c_s[j]]+conv_id[a_s[i]]] = (z_mix_rel_dn_temp, time)
            print(i,j)
	# Store data in json formatted as dictionaries 
    json.dump(K_up, open('K_values_{0}-{1}.txt'.format(160, 600), 'w'))
    json.dump(K_dn, open('K_values_{0}-{1}.txt'.format(600, 920), 'w'))
    json.dump(z_mix_up, open('z_mix_values_{0}-{1}.txt'.format(160, 600), 'w'))
    json.dump(z_mix_dn, open('z_mix_values_{0}-{1}.txt'.format(600, 920), 'w'))
    json.dump(phi_d_up, open('phi_d_values_{0}-{1}.txt'.format(160, 600), 'w'))
    json.dump(phi_d_dn, open('phi_d_values_{0}-{1}.txt'.format(600, 920), 'w'))
    json.dump(Nstar_sq_up, open('Nstar_sq_values_{0}-{1}.txt'.format(160, 600), 'w'))
    json.dump(Nstar_sq_dn, open('Nstar_sq_values_{0}-{1}.txt'.format(600, 920), 'w'))
    json.dump(z_mix_rel_up, open('z_mix_rel_{0}-{1}.txt'.format(160, 600), 'w'))
    json.dump(z_mix_rel_dn, open('z_mix_rel_{0}-{1}.txt'.format(600, 920), 'w'))

def generate_new_set():
	# Generates diapycnal diffusivitiy and zmix json for upstream and downstream
    print('Building json')
    create_jsons()
	

# Run
if __name__ == "__main__":
    generate_new_set()
