import seawater as sw
import numpy as np

T_w = 1 #C
c_p = 4.2 #J/gC
L_T = 3.342 #J/g
C_w = 1 #g/kg
nu = 2e-3 #m^2/s #Viscosity (momentum diffusivity)
kappa = 1.3e-3 #cm^2/s
mu = 2e-3 #m^2/s #Salt mass diffusivity
m = 0.056 #C/(g/kg)

#Parameters defining stratification
scale = -1
z0 = 72
Delta = 1e-1
b = 29

L, H = 960, 80 #m
a, c = 2, 2
l, h = 600, a*(H-z0) #m
sigma = 3.9*h
U = c*np.sqrt((H-z0)*9.8*(sw.dens0(30,-2)-sw.dens0(28,-2))/sw.dens0(28,-2)) #m/s

Re = 1 / nu
Sc = nu / mu #Should be close to 1
S = L_T / (c_p * T_w)
delta = 5e-3
epsilon = 0.125 #Two gridboxes
beta = 4/2.648228 #Not optimized
eta = 1e-3 * Re * (beta * epsilon)**2



#Save parameters
Nx, Nz = 1280, 640
dt = 4e-3 #s #For certain speeds and mixed-layer depths, you can increase this by a factor of 10 to reduce runtime

sim_name = 'mixingsim-Test'
file_handler_mode = 'overwrite'
restart = 0 #Integer
restart_sim = True

steps = 1000 #At 800000 steps, takes many hours to run
save_freq = 150 #15
save_max = 15
print_freq = 100 #Decrease this for diagnostic purposes if the code isn't working
wall_time = 60*60*23
save_dir = '.'
