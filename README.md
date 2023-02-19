# Radiation complex amplitude

This repo provides a python module to compute the instantaneously emitted spectrally and directionally
resolved radiation for a given set of particle positions, momenta and previous momenta. It also supports
particle weight, as commonly used in particle-in-cell codes.


## Usage

````python
# define e.g.
# frequency
omega = np.linspace(1e2, 5e2, 128) # [rad/s]
# observation direction
theta = np.linspace(0, np.pi/2, 16) # [rad]
n = np.array([np.sin(theta), np.cos(theta), np.zeros_like(theta)]).T
# time step duration
delta_t = 1.2e-15 # [s]
# simulation box size
sim_box_size = np.array([100, 100, 100]) # [m, m, m]

# create object
comp_rad = instananeousRadiation(omega, n, delta_t, sim_box_size)

# get your particle data
# ...

# get complex radiation amplitude
rad_amplitude = calc_complexAmplitude_loop(time, # current time in [s]
                                           r, # particle positions in [m, m, m]
					   p, # particle momenta in [kg*m/s, kg*m/s, kg*m/s]
					   p_prev, # particle momenta one time step before in [kg*m/s, kg*m/s, kg*m/s]
					   w) # particle weights

```

