from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..utils.json import read_value_json, update_json


class AutomaticFit:
    """All added variables are processed in the order of creation."""

    activeVar = None
    queue = None
    fitting_mode = False

    def __init__(self, variable, scale_file):
        self.variable = variable  # variable to find value for
        self.scale_file = scale_file

        self._fitted = False
        self.load_maybe()

        # first instance created
        if AutomaticFit.fitting_mode and not self._fitted:
            # if first layer set to active
            if AutomaticFit.activeVar is None:
                AutomaticFit.activeVar = self
                AutomaticFit.queue = []  # initialize
            # else add to queue
            else:
                self._add2queue()

    @staticmethod
    def reset():
        AutomaticFit.activeVar = None
        AutomaticFit.all_processed = False

    @staticmethod
    def fitting_completed():
        return AutomaticFit.queue is None

    @staticmethod
    def set2fitmode():
        AutomaticFit.reset()
        AutomaticFit.fitting_mode = True

    def _add2queue(self):
        logging.debug(f"Add {self._name} to queue.")
        # check that same variable is not added twice
        for var in AutomaticFit.queue:
            if self._name == var._name:
                raise ValueError(f"Variable with the same name ({self._name}) was already added to queue!")
        AutomaticFit.queue += [self]

    def set_next_active(self):
        """Set the next variable in the queue that should be fitted."""
        queue = AutomaticFit.queue
        if len(queue) == 0:
            logging.debug("Processed all variables.")
            AutomaticFit.queue = None
            AutomaticFit.activeVar = None
            return
        AutomaticFit.activeVar = queue.pop(0)

    def load_maybe(self):
        """Load variable from file or set to initial value of the variable."""
        value = read_value_json(self.scale_file, self._name)
        if value is None:
            logging.info(f"Initialize variable {self._name}' to {self.variable.numpy():.3f}")
        else:
            self._fitted = True
            logging.debug(f"Set scale factor {self._name} : {value}")
            with torch.no_grad():
                self.variable.copy_(torch.tensor(value))


class AutoScaleFit(AutomaticFit):
    """Class to automatically fit the scaling factors depending on the observed
    variances.

    Parameters
    ----------
        variable: tf.Variable
            Variable to fit.
        scale_file: str
            Path to the json file where to store/load from the scaling factors.
    """

    def __init__(self, variable, scale_file):
        super().__init__(variable, scale_file)

        if not self._fitted:
            self._init_stats()

    def _init_stats(self):
        self.variance_in = 0
        self.variance_out = 0
        self.nSamples = 0

    def observe(self, x, y):
        """Observe variances for inut x and output y.

        The scaling factor alpha is calculated s.t. Var(alpha * y) ~ Var(x)
        """
        if self._fitted:
            return

        # only track stats for current variable
        if AutomaticFit.activeVar == self:
            nSamples = y.shape[0]
            self.variance_in += torch.mean(torch.var(x, dim=0)) * nSamples
            self.variance_out += torch.mean(torch.var(y, dim=0)) * nSamples
            self.nSamples += nSamples

    def fit(self):
        """Fit the scaling factor based on the observed variances."""
        if AutomaticFit.activeVar == self:
            if self.variance_in == 0:
                raise ValueError(
                    f"Did not track the variable {self._name}. Add observe calls to track the variance before and after."  # noqa: E501
                )

            # calculate variance preserving scaling factor
            self.variance_in = self.variance_in / self.nSamples
            self.variance_out = self.variance_out / self.nSamples

            ratio = self.variance_out / self.variance_in
            value = np.sqrt(1 / ratio, dtype="float32")
            logging.info(
                f"Variable: {self._name}, Var_in: {self.variance_in.numpy():.3f}, Var_out: {self.variance_out.numpy():.3f}, "  # noqa: E501
                + f"Ratio: {ratio:.3f} => Scaling factor: {value:.3f}"
            )

            # set variable to calculated value
            with torch.no_grad():
                self.variable.copy_(self.variable * value)
            update_json(self.scale_file, {self._name: float(self.variable.numpy())})
            self.set_next_active()  # set next variable in queue to active


class ScalingFactor(nn.Module):
    """Scale the output y of the layer s.t. the (mean) variance wrt. to the
    reference input x_ref is preserved.

    Args:
        scale_file (str): Path to the json file where to store/load from the scaling factors.
    """

    def __init__(self, scale_file: str, device=None):
        super().__init__()

        self.scale_factor = torch.nn.Parameter(torch.tensor(1.0, device=device), requires_grad=False)
        self.autofit = AutoScaleFit(self.scale_factor, scale_file)

    def forward(self, x_ref: Tensor, y: Tensor) -> Tensor:
        """
        Returns:
            y (torch.Tensor): scaled ouput.
        """
        y = y * self.scale_factor
        self.autofit.observe(x_ref, y)

        return y
