import numpy as np

class Particle:
    """Localized quantum particle wavefunction, not normalized defined as:

    \Re{\psi} = \exp{ -(-x-t-x0)/(2\sigma^2) }\cos{k_0 x}
    \Im{\psi} = \exp{ -(-x-t-x0)/(2\sigma^2) }\sin{k_0 x}
    \psi^*\psi = \Re{\psi}^2 + \Im{\psi}^2

    Parameters
    ----------
    x0 : float, int
        Initial position of the center of the wavefunction.
    m : float, int
        mass of the particle
    sigma : float, int
        STD of the Gaussian envelope, uncertainty of localization.
    k0 : float, int
        Wavenumber.
    E : float, int
        Energy of the particle, calculated from k.
    """
    def __init__(self, x0, sigma, k0=None, m=1):
        hbar = 1.0e0    #  Plank's constant
        self.x0 = x0
        self.m = m
        self.sigma = sigma
        self.k0 = np.pi/20 if k0 is None else k0
        self.E = (hbar**2/2.0/m)*(k0**2+0.5/sigma**2)

    def real(self, x, t):
        """Real part of the wavefunction."""
        return np.exp(-(x-t-self.x0)**2/(2*self.sigma**2)) * np.cos(self.k0*x)

    def imag(self, x, t):
        """Imaginary part of the wavefunction."""
        return np.exp(-(x-t-self.x0)**2/(2*self.sigma**2)) * np.sin(self.k0*x)

    def prob(self, x, t):
        return self.real(x, t)**2 + self.imag(x, t)**2
