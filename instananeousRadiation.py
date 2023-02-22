import numpy as np
import scipy.constants as const 


class instantaneousRadiation():
    """
    class to compute the spectrally and directionally
    resolved instantaneous radiation for a number of 
    particles
    """
    
    def __init__(self, omega, n, delta_t, sim_box_size):
        """
        initalize the 
            omega ... frequency omega   np.array(N_omega)
            n ... observation direction n   np.array(N_obs, 3)
            delta_t ... time step delta_t   float
            sim_box_size ... simulation box size (in SI units)    np.array(3)
        that will be used for all calculations of the 
        instantaneous radiation
        """
        # TODO: check types
        self.omega = omega  # np.array (1D) or scalar
        self.n = n # np.array  size (N_n, 3)
        self.delta_t = delta_t # float 
        self.sim_box_size = sim_box_size # (float, float, float)
        
        
    def calc_beta(self, p, w):
        """
        compute beta (=v/c) from momenta
        p ... momenta np.array(N_p, 3)
        w ... weight np.array(N_p)
        
        beta = p / (m_e * c * gamma)
        
        return beta np.array(N_p, 3)
        """
        gamma = self.calc_gamma(p, w)
        return (p.T / (const.electron_mass * w * const.speed_of_light * gamma)).T
        
    def calc_gamma(self, p, w):
        """
        compute Lorentz gamma from momenta
        p ... momenta np.array(N_p, 3)
        w ... weight np.array(N_p)
        
        gamma = sqrt( 1 + p^2 / (m_e * c)**2 )
        because
        E^2 = gamma^2 * m_e^2 * c^4 = m_e^2 * c^4 + p^2 * c^2 
        
        return gamma np.array(N_p)
        """
        return np.sqrt( 1 + np.sum(p**2, axis=1) / (w * const.electron_mass * const.speed_of_light)**2 ) 


    def calc_t_ret(self, time, pos):
        """
        compute retarded time t_ret
        time ... current time (scalar)
        pos ... positions np.array(N_p, 3)
        
        t_ret = t - (n * r) / c
        
        return retarded time np.array(N_n, N_p)
        """
        return time - np.sum(self.n[:, np.newaxis, :] * pos[np.newaxis, :, :], axis=-1) / const.speed_of_light
    
    
    def calc_real_amplitude(self, p, p_prev, w):
        """
        compute real vector amplitude 
        p ... momenta np.array(N_p, 3)
        p_prev ... previous momenta np.array(N_p, 3)        
        w ... weight np.array(N_p)
        
        computes real amplitue see eq. 4.4 PhD thesis R. Pausch (real vector part)
        
        return (real vector amplitude np.array(N_n, N_p, 3), 
                one minus n times beta np.array(N_n, N_p)) 
        """
        beta = self.calc_beta(p, w)
        beta_dot = (beta - self.calc_beta(p_prev, w)) / self.delta_t

        # n - beta: [N_n, N_p, 3]
        n_minus_beta = (self.n[:, np.newaxis, :] - beta[np.newaxis, :, :])
        
        # (n - beta) % dot(beta): [N_n, N_p, 3]
        n_minus_beta_cross_betaDot = np.cross(n_minus_beta, beta_dot)
        del(n_minus_beta)
        
        # n % [ (n - beta) % dot(beta) ]: [N_n, N_p, 3] 
        non_normed_amplitude = np.cross(self.n[:, np.newaxis, :], n_minus_beta_cross_betaDot)
        del(n_minus_beta_cross_betaDot)
        
        # 1 - n * beta: [N_n, N_p]
        One_minus_beta_times_n = ((1.0 - np.sum(self.n[:, np.newaxis, :] * beta[np.newaxis, :, :], axis=-1)))
        
        # { n % [ (n - beta) % dot(beta) ]} / {1 - n * beta}^2 and (1 - n * beta) to not run this computation again
        return non_normed_amplitude * One_minus_beta_times_n[:, :, np.newaxis]**2, One_minus_beta_times_n
    
    
    def check_Nyquist(self, One_minus_beta_times_n, NyquistFactor=0.5):
        """
        return true or false whether Nyquist limit has not been reached yet
        One_minus_beta_times_n ... 1 - beta*n (precomputed during previous step) np.array(N_n, N_p)
        NyquistFactor ... aditional reduction factor <= 1 to avoid hitting Nyquist frequency (scalar) [default: 0.5]
        
        omega < NyquistFactor * pi / (delta_t * (1- beta * n))
        
        return boolean map of wheter resuts makes sense (1) or not (0) np.array(N_omega, N_n, N_p)
        """
        omegaNyquist = (np.pi * 0.99) / (self.delta_t * One_minus_beta_times_n)
        return np.less(self.omega[:, np.newaxis, np.newaxis], omegaNyquist[np.newaxis, :, :] * NyquistFactor)
    
    def calc_window_function(self, pos):
        """
        compute the Triplett window function for the simulation box
        please be aware that windows can be changed - this is the window currently used
        
        pos ... position np.array(N_p, 3)
        
        exp(-lambda * abs(x)) * [ cos(pi * x / L) |^2
        x in -L/2 till +L/2
        lambda just needs to be "large enough" to avoid side lobes 
        
        return window value for each particle np.array(N_p)
        """
        # TODO: hard coded a Triplett window - needs flexibility 
        x = pos - 0.5 * self.sim_box_size
        # TODO: here seems to be an error in the PIConGPU code (just kept it this wrong way - might need fix soon)
        cosinusValue = np.cos(np.pi*(x / (5.0 / self.sim_box_size)))

        return np.sum(np.less(np.abs(x), 0.5 * self.sim_box_size) * # test if particle is in sim box
                np.exp(-1. * (5.0 / self.sim_box_size) * np.abs(x)) * cosinusValue**2, axis=-1) # shape of tripplet window
    
    def calc_radFormFactor_Gauss_spherical(self, w):
        """
        compute the frequency dependent form facor of a macro particle
        this is neeed to get the right scaling for coherent and incoherent radiaton since a macro-particle describes multiple electrons
        
        w ... weighting np.array(N_p)
        
        form factor = N + (N^2-N) * Fourtier Transfor of assumed shape 
        we assume a Gaussian shape (not the shape the PIC code assumes)
        
        return sqrt(form factor) (will to be squared later) np.array(N_omega, N_p)
        """
        return np.sqrt(w[np.newaxis, :] + (w**2 - w)[np.newaxis, :] * (np.exp(-0.5 * (0.5 * self.omega[:, np.newaxis] * self.delta_t)**2))**2 )

    
    def calc_complexAmplitude(self, time, pos, p, p_prev, w):
        """
        compute the complex radiation amplitude \mathbb{C}^3 for the given particles
        
        time ... current time (float) in SI units
        pos ... position np.array(N_p, 3) in SI units
        p ... momentum np.array(N_p, 3) in SI units
        p_prev ... momentum of previous time step np.array(N_p, 3) in SI units
        w ... weight np.array(N_p)
        
        return complex radiation amplitude np.array(N_omega, N_n, {x,y,z}) dtype=complex
        """
        # [n, particles, {x,y,z}]
        real_amplitude, One_minus_beta_times_n = self.calc_real_amplitude(p, p_prev, w)
        # [omega, n, particle]
        complex_one = (np.exp(1j * self.omega[:, np.newaxis, np.newaxis] * self.calc_t_ret(0.0, pos)[np.newaxis, :, :]) 
                       * self.check_Nyquist(One_minus_beta_times_n)
                       * self.calc_window_function(pos)[np.newaxis, np.newaxis, :]
                       * self.calc_radFormFactor_Gauss_spherical(w)[:, np.newaxis, :]
            )
        
        # [omega, n, particles, {x,y,z}]
        return np.sum(real_amplitude[np.newaxis, :, :, :] * complex_one[:, :, :, np.newaxis], axis=-2)    
    
    def calc_complexAmplitude_loop(self, time, pos, p, p_prev, w, step_width = 128):
        """
        compute the complex radiation amplitude \mathbb{C}^3 for the given particles
        this method uses internal looping to keep the memeory profile low
        --> please use this method 
        
        time ... current time (float) in SI units
        pos ... position np.array(N_p, 3) in SI units
        p ... momentum np.array(N_p, 3) in SI units
        p_prev ... momentum of previous time step np.array(N_p, 3) in SI units
        w ... weight np.array(N_p)
        
        return complex radiation amplitude np.array(N_omega, N_n, {x,y,z}) dtype=complex
        """
        complex_amplitude = np.zeros((len(self.omega), len(self.n), 3), dtype=np.complex128)
        for index in np.arange(0, len(pos), step_width):
            complex_amplitude += self.calc_complexAmplitude(time, 
                                                            pos[index:index+step_width], 
                                                            p[index:index+step_width], 
                                                            p_prev[index:index+step_width], 
                                                            w[index:index+step_width])
            
        return complex_amplitude




if __name__ == "__main__":
    # setup observer space

    # time step
    delta_t = 1.2
    
    # frequency
    omega = np.linspace(1e2, 5e2, 128)

    # observation direction
    theta = np.linspace(0, np.pi/2, 16)
    n = np.array([np.sin(theta), np.cos(theta), np.zeros_like(theta)]).T
    
    # simulation box size
    sim_box_size = np.array([100, 100, 100])
    
    
    # create object
    inst_rad_ampl = instantaneousRadiation(omega, n, delta_t, sim_box_size)
    
    # create non-sense particle data
    # number of particles:
    N_p = 100000

    # position
    r = np.zeros((N_p , 3))
    r[:, 1] = np.random.normal(size=N_p)
    
    # momenta
    p = np.zeros((N_p , 3))
    v = 1.0 - np.logspace(0.0, -10, N_p)
    p_x = v  * const.c * const.electron_mass *2* np.sqrt(1./(1- v**2 ))
    p[:, 1] = p_x

    # previous momenta
    p_prev = np.zeros((N_p , 3))
    p_prev[:, 1] = p_x * 0.95
    p_prev[:, 2] = p_x * 0.05

    # weighting
    w = 2*np.ones(N_p)
    
    # compute instantaneous amplitude
    complex_amplitude = inst_rad_ampl.calc_complexAmplitude(1.2, r, p, p_prev, w)

    print(complex_amplitude)
