U = 0.4 #m/s
T_w = 1 #C
c_p = 4.2 #J/gC
L_T = 3.342 #J/g
C_w = 1 #g/kg
nu = 5e-5 #m^2/s #Viscosity (momentum diffusivity)
kappa = 1.3e-3 #cm^2/s
mu = 5e-5 #m^2/s #Salt mass diffusivity
m = 0.056 #C/(g/kg)
L, H = 110, 34 #m
l, h = 70, 10-4 #m

Re = 1 / nu
Sc = nu / mu #Should be close to 1
S = L_T / (c_p * T_w)
delta = 5e-3
epsilon = 0.125 #Two gridboxes
beta = 4/2.648228 #Not optimized
eta = 1e-5 * Re * (beta * epsilon)**2

#Parameters defining stratification
scale = -1
z0 = 19
Delta = 1e-1
b = 27

#Save parameters
Nx, Nz = 640, 640
dt = 5e-4 #s #For certain speeds and mixed-layer depths, you can increase this by a factor of 10 to reduce runtime

sim_name = 'mixingsim-Test'
restart = 0 #Integer

steps = 2000 #At 800000 steps, takes many hours to run
save_freq = 45 #15
save_max = 15
print_freq = 5000 #Decrease this for diagnostic purposes if the code isn't working
wall_time = 60*60*23
save_dir = '.'