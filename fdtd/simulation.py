import numpy as np

class Simulation:
    """Solves the equations FDTD update equations for provided potentials and
    a particle.

    Parameters
    ----------
    particle : Particle
        Particle object used in the simulation
    potentials : list, tuple
        Potentials used in the simulation, end potential is constructed by
        evaluating and then adding all used potentials.
    N : int
        Number of spatial steps used, defaults to 1200
    dx : float, int
        Spatial resolution of simulation, defaults to 1.0
    xlims : list, tuple
        Spatial left and right limits, together with N and dx used to define the
        spatial scale on which the simulation runs
    T : float, int
        End time of the simulation, defaults to 5N
    dt : float, int
        Temporal resolution, very sensitive and best left unset since then it
        defaults to hbar/( 2*hbar**2/(self.p.m*self.dx**2)+V )
    """
    def __init__(self, particle, potentials, N=1200, dx=1., xlims=None,
                 T=None, dt=None):
        self.N = N
        self.T = 5*N if T is None else T
        self.dx= dx
        self.dt = dt
        startx = 0 if xlims is None else xlims[0]
        endx = N*dx if xlims is None else xlims[1]
        self.scale = np.arange(startx, endx, dx)

        self.potentials = potentials
        self.V = self.evalPotentials()
        self.p = particle

    def evalPotentials(self):
        """Evaluates the list of potentials to produce on the scale."""
        V = np.zeros((self.N, ))
        for pot in self.potentials:
            V += pot.f(self.scale)
        return V

    def simulate(self, deltaT):
        """Simulation is a generator that returns total probability and its
        real and imaginary parts, normalized, every deltaT-th step.
        """
        # often used, cleans up the code.
        N = self.N
        hbar = 1.0e0

        # time step is critical for numerical stability
        if self.dt is None:
            dt = hbar / ( 2*hbar**2/(self.p.m*self.dx**2) + self.V.max() )
        else:
            dt = self.dt

        #  Wave functions. Real and imaginary are actually 3 arrays of same
        # length representing hree states represent past, present, and future.
        psi_r = np.zeros((3,N))
        psi_i = np.zeros((3,N))
        psi_p = np.zeros(N,)

        # past, present and future indices to be used instead of 0, 1, 2
        # to make it clear contextually what's happening
        PA, PR, FU = 0, 1, 2

        psi_r[PR] = self.p.real(self.scale, 0)
        psi_i[PR] = self.p.imag(self.scale, 0)
        psi_r[PA] = self.p.real(self.scale, 0)
        psi_i[PA] = self.p.imag(self.scale, 0)
        psi_p = self.p.prob(self.scale, 0)

        #  Normalize the wave functions
        P   = self.dx * psi_p.sum()
        nrm = np.sqrt(P)
        psi_r /= nrm
        psi_i /= nrm
        psi_p /= P

        # constants from the derivation
        c1   = hbar*dt/(self.p.m * self.dx**2)
        c2   = 2*dt/hbar
        c2V  = c2*self.V

        # Direct index assignment is faster than using a spatial for loop
        # Precomputing k, k+1, k-1 is also easy, so we do that instead
        IDX1 = np.arange(1, N-1, 1)
        IDX2 = np.arange(2, N, 1)
        IDX3 = np.arange(0, N-2, 1)
        for t in range(self.T+1):
            psi_rPR = psi_r[PR]
            psi_iPR = psi_i[PR]

            #  update equations
            psi_i[FU,IDX1] = psi_i[PA,IDX1] +  \
                c1*(psi_rPR[IDX2] - 2*psi_rPR[IDX1] +  psi_rPR[IDX3])
            psi_i[FU] -= c2V*psi_r[PR]

            psi_r[FU,IDX1] = psi_r[PA,IDX1] - \
                c1*(psi_iPR[IDX2] - 2*psi_iPR[IDX1] +  psi_iPR[IDX3])
            psi_r[FU] += c2V*psi_i[PR]

            #  PR -> PA and FU -> PR
            psi_r[PA] = psi_rPR
            psi_r[PR] = psi_r[FU]
            psi_i[PA] = psi_iPR
            psi_i[PR] = psi_i[FU]
            #  Only plot after a few iterations to make the simulation run faster.
            if t % deltaT == 0:
                psi_p = psi_r[PR]**2 + psi_i[PR]**2
                yield {'prob': psi_p, 'real':psi_r[PR], 'imag':psi_i[PR]}
