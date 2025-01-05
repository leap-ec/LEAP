#!/usr/bin/env python3
""" Visualization pipeline operators tailored for multiple objectives
"""
import numpy as np
from matplotlib import pyplot as plt

from leap_ec.util import get_step
from leap_ec.probe import PopulationMetricsPlotProbe
from leap_ec.global_vars import context



##############################
# Class ParetoPlotProbe2D
##############################
class ParetoPlotProbe2D(PopulationMetricsPlotProbe):
    """
    Plot a 2D Pareto front of a population that has been assigned
    multi-objective fitness values.

    If the fitness space has more than two dimensions, only the first two are
    plotted.
    """
    def __init__(self, ax=None, metrics=None, xlim=(0, 1), ylim=(0, 1),
                 title='Pareto Front',
                 step=1, context=context):
        """
        :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will
            be created).
        :param xlim: Bounds of the horizontal axis.
        :type xlim: (float, float)
        :param ylim: Bounds of the vertical axis.
        :type ylim: (float, float)
        :param int step: take and plot a measurement every `step` steps (
            default 1).
        :param title: title to print on the plot
        :param context: set a context object to query for the current generation.
            Defaults to the standard `leap_ec.context` object.
        """
        super().__init__(ax=ax, metrics=metrics, xlim=xlim, ylim=ylim,
                         title=title, modulo=step,
                         context=context)

        self.scatterplot = ax.scatter([], []) # scatterplot for fitnesses

    def __call__(self, population):
        assert (population is not None)
        step = get_step(self.context)

        if step % self.modulo == 0:
            self.x = np.array([ind.fitness[0] for ind in population])
            self.y = np.array([ind.fitness[1] for ind in population])
            self.scatterplot.set_offsets(np.c_[self.x, self.y])
            self._rescale_ax()
            self.ax.figure.canvas.draw()
            # With previous versions of matplotlib and/or jupyter, this line was needed to
            #   trigger the plot to update in real time.  But as of the time of this comment,
            #   it is either not needed or caused problems.
            #plt.pause(0.000001)

        return population

    def reset(self):
        self.x = np.array([])
        self.y = np.array([])




if __name__ == '__main__':
    probe = ParetoPlotProbe2D()

    pass
