import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class AnimateSim(animation.FuncAnimation):
    """Animate a FDTD simulation with this class.

    Parameters
    ----------
    simulation : Simulation
        Simulation object that will be animated.
    deltaT : int
        Every deltaT-th step will be plotted as part of the animation
    plot_prob : bool
        if False total probability will not be plotted
    plot_real : bool
        if False the Re part of the particle's wavefunction will not be plotted
    plot_imag : bool
        if False the Im part of the particle's wavefunction will not be plotted
    plot_pot : bool
        if False the area under the potential will not be shaded
    **kwargs : dict
        any additional kwargs are forwarded to FuncAnim from matplotlib
    """
    def __init__(self, simulation, **kwargs):
        self.sim = simulation
        self.steps = simulation.simulate(kwargs.pop('deltaT', 50))

        self.kwargs = kwargs
        self.update = {
            'prob': kwargs.pop('plot_prob', True),
            'real': kwargs.pop('plot_real', True),
            'imag': kwargs.pop('plot_imag', True)
        }

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        super().__init__(self.fig, self.stepAnim, init_func=self.initAnim,
                         **kwargs)

    def initAnim(self):
        """Determines the general layout of the plot and draws the first sim
        step.
        """
        step = next(self.steps)
        makeFuncAnimHappy = []

        # potentials need not be updated with time but can be plotted
        if self.kwargs.pop('plot_pot', True):
            self.ax.fill_between(self.sim.scale, self.sim.V, facecolor='y',
                                 alpha=0.2, label='Potential')

        if self.update['real']:
            self.update['real'], = self.ax.plot(self.sim.scale, step['real'],
                                                'r', label='Real')
        if self.update['imag']:
            self.update['imag'], = self.ax.plot(self.sim.scale, step['imag'],
                                                'b', label='Imag')
        if self.update['prob']:
            self.update['prob'], = self.ax.plot(self.sim.scale, step['prob'],
                                                'k', label='Prob')

        miny = - max(step['real'])/10
        maxy = max(step['imag']) - miny
        self.ax.set_ylim(-maxy, maxy)
        plt.legend()
        return self.update.values()

    def stepAnim(self, i):
        """Makes an animation step and updates the plot."""
        step = next(self.steps)
        for key, val in self.update.items():
            self.update[key].set_ydata(step[key])
        return self.update.values()
