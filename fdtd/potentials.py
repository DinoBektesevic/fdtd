import numpy as np
from scipy import interpolate
import warnings

__all__ = ['Potential', 'NullPotential', 'PointPotential', 'BarrierPotential', 'StepPotential']

class Potential:
    """Generic potential class. Expects to be provided a name of the potential.
    Inheritted objects need to define the potential functions _fx and _farr
    that return the potential values in a point or on an array.
    """
    def __init__(self, name='generic', *args, **kwargs):
        self.name = name

    def f(self, x):
        if hasattr(x, '__iter__'):
            return self._farr(x)
        else:
            return self._fx(x)

    def _fx(self, x):
        raise NotImplemented('Method not declared in inherrited class!')

    def _farr(self, x):
        return NotImplemented('Method not declared in inherrited class!')


class NullPotential(Potential):
    """Pontential of a free particle. Identically zero everywhere."""
    def __init__(self):
        super().__init__(name='Free particle')
        self.A = 0

    def _fx(self, x):
        return 0

    def _farr(self, x):
        return np.zeros(len(x), dtype=float)

class PointPotential(Potential):
    """Potential with amplitude A in a single point and zero everywhere else.

    Parameters
    ----------
    A : float, int
        Amplitude
    pos : float, int
        Position of the peak amplitude, defaults to 0
    tolerance: float, int
       Evaluating point potential on a discrete grid can lead to situations
       where none of the discrete coordinates are neccessarily close enough to
       position. Using poor resolution can lead to point potential to look like
       barrier. Tolerance defines the allowable range of values where 'point'
       is defined.
    """
    def __init__(self, A, pos=None, tolerance=None):
        super().__init__(name='Point potential')

        if pos is None:
            pos = 0
        if tolerance is None:
            tolerance = 0.0001

        self.A = A
        self.pos = pos
        self.epsilon = tolerance

    def _fx(self, x):
        if abs(x) <= (self.pos+self.tolerance):
            return self.A
        else:
            return 0

    def _farr(self, x):
        res = np.zeros(len(x))

        if (min(x) > self.pos) or (max(x) < self.pos):
            warnings.warn('Scale does not seem to contain the potential.')
            return res

        shiftx = abs(x - self.pos)
        idx = np.where(shiftx == min(shiftx))[0][0]


        if abs(x[idx]) > abs(self.pos + self.epsilon):
            warnings.warn((f'Closest position on the scale is {x[idx]},'
                           f'which is outside of allowed tolerance {self.pos} +/- {self.epsilon}'))
            return res

        res[idx] = self.A
        return res

class BarrierPotential(Potential):
    """Barrier potential. Has amplitude A across the width [pos-width, pos+width]
    and zero elsewhere.

    Parameters
    ----------
    A : float, int
        Amplitude.
    pos : float, int
        Position of barrier center-point, defaults to 0.
    width : float, int
        Width of the barrier.
    """
    def __init__(self, A, width, pos=None):
        super().__init__(name='Barrier potential')

        if pos is None:
            pos = 0

        self.A = A
        self.width = width
        self.pos = pos

    def _fx(self, x):
        if abs(x) <= abs(self.pos + self.width):
            return self.A
        else:
            return 0

    def _farr(self, x):
        res = np.zeros(len(x))
        res[ (x >= self.pos) & (x <= self.pos+self.width)] = self.A
        res[ (x <  self.pos) & (x >= self.pos-self.width)] = self.A
        return res

class StepPotential(Potential):
    """Step potential. Has amplitude A in the range [pos, inf.> and is zero in
    range <-inf., pos].

    Parameters
    ----------
    A : float, int
        Amplitude.
    pos : float, int
        Position of the step, defaults to 0.
    """
    def __init__(self, A, pos=None):
        super().__init__(name='Step potential')

        if pos is None:
            pos = 0

        self.A = A
        self.pos = pos

    def _fx(self, x):
        if x < self.pos:
            return self.A
        return 0

    def _farr(self, x):
        res = np.zeros(len(x))
        res[x >= self.pos] = self.A
        return res
