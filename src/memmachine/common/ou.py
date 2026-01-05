import numpy as np
from datetime import datetime, timedelta


class OrnsteinUhlenbeckRecruiter:
    def __init__(
        self,
        num_values: int,
        num_recruits: int,
        mu: float = 0.0,
        sigma: float = 0.004,
        theta: float = 0.00003,
        beta: float = 6.0,
    ):
        self._num_values = num_values
        self._num_recruits = num_recruits

        self._mu = np.full(self._num_values, mu)
        self._sigma = sigma
        self._theta = theta

        self._beta = beta

        self._excitability = np.full(num_values, self._mu)
        self._latest_time = None

    def update_states(self, new_time: datetime):
        if self._latest_time is None:
            self._latest_time = new_time
            return

        dt = (new_time - self._latest_time).total_seconds()
        if dt <= 0:
            return

        exp_theta = np.exp(-self._theta * dt)
        mean = self._excitability * exp_theta + self._mu * (1 - exp_theta)
        exact_std = np.sqrt(
            (self._sigma**2 / (2 * self._theta)) * (1 - np.exp(-2 * self._theta * dt))
        )
        noise = exact_std * np.random.normal(size=self._num_values)
        self._excitability = mean + noise
        self._latest_time = new_time

    def recruit_indices(self, new_time: datetime):
        self.update_states(new_time)

        shifted_excitability = self._excitability - np.max(self._excitability)
        exp_excitability = np.exp(self._beta * shifted_excitability)
        probabilities = exp_excitability / exp_excitability.sum()

        indices = np.random.choice(
            self._num_values, size=self._num_recruits, replace=False, p=probabilities
        )

        self._excitability[indices] = 1.0
        return indices
