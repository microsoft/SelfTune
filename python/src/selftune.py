"""
This module implements SelfTune, a framework that uses data-driven tuning of
parameters that produce larger rewards.
"""
import math
from typing import List, Union, Tuple, Optional
import numpy as np


class Constraints:
    """A class to hold constraints on a parameter learned using SelfTune.

    Attributes:
        min: The minimum value a parameter can take.
        max: The maximum value a parameter can take.
        is_int: Specifies whether a parameter is an integer.
    """
    DEFAULT_MIN = 2.2250738585072014E-308  # 64 bit floating point min value
    DEFAULT_MAX = 1.7976931348623158E+308  # 64 bit floating point max value

    def __init__(self,
                 c_min: Union[int, float] = DEFAULT_MIN,
                 c_max: Union[int, float] = DEFAULT_MAX,
                 is_int: bool = False):
        self.min = c_min
        self.max = c_max
        self.is_int = is_int


class TaskData:
    """A helper class that holds data representing the learning task.

    Attributes:
        initial_values: Initial values for the parameters.
        constraints: Constraints on the parameters.
        opt: The optimization algorithm to use. Can take values {"bluefin"}.
        feedback: Type of feedback update. Can take values
            {"onepoint", "twopoint"}.
        eta: The value of the learning rate in the ongoing round.
        initial_eta: The initial value of eta.
        delta: The exploration radius.
        eta_decay_rate: The decay rate of eta.
        is_ints: A numpy array of the is_int constraint for all parameters.
            Used to optimize computations.
        feedback_denom: Used in the gradient estimation process. The value is
            1 for onepoint and 2 for twopoint.
    """

    def __init__(self, initial_values: np.ndarray,
                 constraints: Optional[List[Constraints]], opt: str,
                 feedback: str, eta: float, delta: float,
                 eta_decay_rate: float):
        self.initial_values = initial_values
        self.constraints = constraints
        self.opt = opt
        self.feedback = feedback
        self.eta = eta
        self.initial_eta = eta
        self.delta = delta
        self.eta_decay_rate = eta_decay_rate
        self.is_ints = np.asarray([i.is_int for i in constraints])

        # Used during gradient estimation
        self.feedback_denom = 1 if feedback == 'onepoint' else 2

    def validate_params(self):
        """Validates the task data."""
        n_parameters = self.initial_values.size

        assert len(self.initial_values.shape) == 1

        if self.constraints is not None:
            assert len(self.constraints) == n_parameters, \
                'Constraints not provided for all parameters'

        for constraint in self.constraints:
            assert constraint.min < constraint.max, \
                f'Invalid constraint: Min={constraint.min} >= ' \
                f'Max={constraint.max}'

        assert self.delta > 0, f'Delta={self.delta} should be > 0'


class SessionData:
    """A helper class that holds data representing the session.

    Attributes:
        first_explore: The first point to explore as part of this session.
        second_explore: The second point to explore as part of this session.
        center: The point around which exploration is done.
        explore_direction: Direction along which exploration is done.
        first_explore_id: Identifier for first_explore.
        second_explore_id: Identifier for second_explore.
        first_reward: Used to store the reward from the first explore during
            two-point feedback.
    """

    def __init__(self, center: np.ndarray, explore_direction: np.ndarray,
                 taskdata: TaskData, first_id: int):
        # Explore in the direction of explore_direction
        self.first_explore = SelfTune.compute_explore_value(
            center, 1, explore_direction, taskdata)
        # Explore in the direction opposite to explore_direction
        self.second_explore = None if taskdata.feedback == 'onepoint' else \
            SelfTune.compute_explore_value(
                center, -1, explore_direction, taskdata)
        self.center = center
        self.explore_direction = explore_direction
        self.first_explore_id = first_id
        self.second_explore_id = 0 if taskdata.feedback == 'onepoint' else first_id + 1
        self.first_reward = None


class SelfTune:
    """SelfTune Class.

    Attributes:
        initial_values: Initial values for the parameters.
        constraints: Constraints on the parameters.
        opt: The optimization algorithm to use. Can take values {'bluefin'}.
        feedback: Type of feedback update. Can take values
            {'onepoint', 'twopoint'}.
        eta: The learning rate.
        delta: The exploration radius.
        random_state: The random seed used to initialize the numpy
            pseudo-random number generator.
        eta_decay_rate: The decay rate of eta. Defaults to 0 where eta
            does not decay after each round.
        taskdata: Stores data relating to the learning task.
        rs: The random number generator used by the algorithm.
    """

    def __init__(self,
                 initial_values: np.ndarray,
                 constraints: Optional[List[Constraints]] = None,
                 opt: str = 'bluefin',
                 feedback: str = 'onepoint',
                 eta: float = 0.01,
                 delta: float = 0.1,
                 random_state: int = None,
                 eta_decay_rate: float = 0):
        self.round = 0

        self.taskdata = TaskData(initial_values, constraints, opt, feedback,
                                 eta, delta, eta_decay_rate)
        self.taskdata.validate_params()

        self.random_state = np.random.RandomState(seed=random_state)

        if self.taskdata.constraints is not None:
            initial_values = SelfTune.project(initial_values,
                                              self.taskdata.constraints,
                                              self.taskdata.delta)

        self.session = SessionData(
            initial_values, self.sample_unit_sphere(initial_values.shape),
            self.taskdata, 0)

    def predict(self, features=None) -> np.ndarray:
        """Gets the next set of parameter values to use.

        Args:
            features: features to use for the prediction. The bluefin
                algorithm does not use any features.

        Returns:
            A numpy array of the parameter values to use.
        """
        if self.session.first_reward is None:
            return self.session.first_explore

        return self.session.second_explore

    def set_reward(self, reward: float):
        """
        Sets the reward based on the parameters returned from predict.

        Args:
            reward: A measure of how well the parameter values returned in
                predict performed.
        """
        if self.taskdata.feedback_denom == 1:  # onepoint
            new_center = SelfTune.compute_new_center(
                self.session.center, self.session.explore_direction, reward,
                self.taskdata)
            self.session = SessionData(
                new_center, self.sample_unit_sphere(new_center.shape),
                self.taskdata, self.session.first_explore + 1)
            self.round += 1
            self.taskdata.eta = self.eta_decay()  # Eta Decay

        else:  # twopoint
            if self.session.first_reward is None:
                self.session.first_reward = reward
            else:
                reward_diff = (self.session.first_reward - reward)

                # Normalizing reward diff
                reward_diff /= max(abs(self.session.first_reward), abs(reward))

                new_center = SelfTune.compute_new_center(
                    self.session.center, self.session.explore_direction,
                    reward_diff, self.taskdata)
                self.session = SessionData(
                    new_center, self.sample_unit_sphere(new_center.shape),
                    self.taskdata, self.session.second_explore + 1)
                self.round += 1
                self.taskdata.eta = self.eta_decay()  # Eta Decay

    def sample_unit_sphere(self, shape: Tuple) -> np.ndarray:
        """Sample uniformly from the surface of a sphere.

        Args:
            shape: The shape of the returned sample.

        Returns:
            A vector with values sampled from the surface of a sphere in shape
            dimensions.
        """
        # Sample from a standard normal distribution i.e ~ N(0, 1).
        sample = self.random_state.normal(size=shape, loc=0, scale=1)
        norm = np.linalg.norm(sample)

        if norm == 0:  # norm == 0 is very unlikely, nevertheless handling it
            sample[0] = self.random_state.randint(0, 2) * 2 - 1
        else:
            sample /= np.linalg.norm(sample)

        # print(f'sample.Add(new List<double>{{{sample[0]}, {sample[1]}}});')
        return sample

    def eta_decay(self):
        """Decay eta based on the decay rate and current round.

        Returns:
            The new value of eta.
        """
        return self.taskdata.initial_eta / (
            1 + self.taskdata.eta_decay_rate * self.round)

    @property
    def center(self) -> np.ndarray:
        return self.session.center

    @property
    def predict_id(self) -> int:
        return self.session.first_explore_id if self.session.first_reward is None else self.session.second_explore_id

    @property
    def eta(self) -> float:
        return self.taskdata.eta

    @eta.setter
    def eta(self, e):
        assert e > 0, f'Eta={e} should be > 0'
        self.taskdata.eta = e

    @property
    def delta(self) -> float:
        return self.taskdata.delta

    @delta.setter
    def delta(self, d):
        assert d > 0, f'Delta={d} should be > 0'
        self.taskdata.delta = d

    @staticmethod
    def compute_new_center(center: np.ndarray, explore_direction: np.ndarray,
                           reward_grad: float,
                           taskdata: TaskData) -> np.ndarray:
        """Computes the updated set of parameters.

        Computes the updated set of parameters using the reward(s) obtained
        during the exploration around the current set of parameters.

        Args:
            center: The current parameter vector.
            explore_direction: Direction of exploration.
            reward_grad: Contribution of the reward(s) toward the gradient.
            taskdata: Task for which new center is to be computed.

        Returns:
            New set of parameters.
        """
        new_center = SelfTune.project(
            center +
            (taskdata.eta * center.size * reward_grad * explore_direction) /
            (taskdata.feedback_denom * taskdata.delta), taskdata.constraints,
            taskdata.delta)

        return new_center

    @staticmethod
    def compute_explore_value(center: np.ndarray, explore_sign: int,
                              explore_direction: np.ndarray,
                              taskdata: TaskData) -> np.ndarray:
        """Compute the point to explore.

        Computes the point to explore by randomly perturbing the center
        vector. The explore vector is sampled from a hypersphere centered at
        the center vector with a radius of delta.

        Args:
            center: The point around which exploration is done.
            explore_sign: Specifies whether exploration is in the direction of
                explore_dir or opposite to it.
            explore_direction: Direction of exploration.
            taskdata: Task for which the point to explore is computed.

        Returns:
            A numpy array that represents the point to explore.
        """
        change = explore_sign * taskdata.delta * explore_direction

        if taskdata.is_ints.any():
            rounded = np.round(change)
            change = np.where((rounded != 0) & (taskdata.is_ints), rounded,
                              change)
            change = np.where(
                (rounded == 0) & (change >= 0) & (taskdata.is_ints), 1, change)
            change = np.where(
                (rounded == 0) & (change < 0) & (taskdata.is_ints), -1, change)

        explore_value = center + change
        return explore_value

    @staticmethod
    def project(center: np.ndarray, constraints: List[Constraints],
                delta: float) -> np.ndarray:
        """Projects center vector based on constraints.

        Args:
            center: The vector of parameters to project.
            constraints: Constraints on the parameters.
            delta: The exploration radius of the learning task.


        Returns:
            The projected vector.
        """
        new_center = []
        for idx, val in enumerate(center):
            new_center.append(
                SelfTune.project_single(val, constraints[idx], delta))

        return np.array(new_center)

    @staticmethod
    def project_single(val: float, constraint: Constraints,
                       delta: float) -> float:
        """
        Projects a single value based on constraints.

        Args:
            val: The value to project.
            constraint: Constraints on the value.
            delta: The exploration radius of the learning task.

        Returns:
            The projected value.
        """
        is_int = constraint.is_int
        c_min = constraint.min
        c_max = constraint.max
        if is_int:
            c_min = math.ceil(c_min)
            c_max = math.floor(c_max)
            delta = math.ceil(delta)

        c_min += delta
        c_max -= delta

        new_val = min(max(c_min, val), c_max)
        if is_int:
            new_val = np.round(new_val)

        return new_val
