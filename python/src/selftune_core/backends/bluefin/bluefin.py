"""
This module implements BlueFin, a framework that uses data-driven tuning of
parameters that produce larger rewards.
"""
import logging
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from ...utils.normalizers.min_max import min_max_denorm, min_max_norm
from ...utils.optimizers import SGD, RMSprop
from ...utils.parameters import Constraint, ContinuousParameter, DiscreteParameter
from ...utils.selftune_logging import get_logger, log_args_and_return

_logger = get_logger(logger_name=__name__)


class TaskData:
    """A helper class that holds data representing the learning task.

    Attributes:
        param_names: Names of the parameters.
        initial_values: Initial values for the parameters.
        constraints: Constraints on the parameters.
        algorithm: The optimization algorithm to use. Can take values {'bluefin'}.
        feedback: Type of feedback update. Can take values {"onepoint", "twopoint"}.
        eta: The value of the learning rate in the ongoing round.
        initial_eta: The initial value of eta.
        delta: The exploration radius.
        optimizer: The optimizer to use.
        optimizer_kwargs: The optimizer keyword arguments.
        eta_decay_rate: The decay rate of eta.
        normalize: Specifies whether the parameter values have to be normalized. Uses min-max normalization.
        lbs: A numpy array of the min constraint for all parameters. Used to optimize computations.
        ubs: A numpy array of the max constraint for all parameters. Used to optimize computations.
        is_ints: A numpy array of the is_int constraint for all parameters. Used to optimize computations.
        feedback_denom: Used in the gradient estimation process. The value is 1 for onepoint and 2 for twopoint.
    """

    _FEEDBACK_VALUES = ("onepoint", "twopoint")
    _OPTIMIZERS = ("sgd", "rmsprop")

    @log_args_and_return(logger=_logger)
    def __init__(
        self,
        param_names: Iterable[str],
        initial_values: np.ndarray,
        constraints: Optional[Iterable[Constraint]],
        algorithm: str,
        feedback: str,
        eta: float,
        delta: float,
        optimizer: str,
        optimizer_kwargs: dict,
        eta_decay_rate: float,
        normalize: bool,
    ):
        self.param_names = tuple(param_names)
        self.initial_values = initial_values
        self.constraints = tuple(constraints)  # TODO Add check for constraints = None
        self.normalized_constraints = None
        self.algorithm = algorithm
        self.feedback = feedback
        self.eta = eta
        self.initial_eta = eta
        self.delta = delta
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_kwargs["eta"] = self.initial_eta
        self.eta_decay_rate = eta_decay_rate
        self.normalize = normalize
        self.normalized_constraints = BlueFin.get_normalized_constraints(len(self.param_names))
        self.lbs = np.asarray([c.lb for c in constraints])  # TODO Add check for constraints = None
        self.ubs = np.asarray([c.ub for c in constraints])
        self.is_ints = tuple(c.is_int for c in constraints)

        # Used during gradient estimation
        self.feedback_denom = 1 if feedback == "onepoint" else 2

        self.validate_taskdata()

    @log_args_and_return(logger=_logger)
    def validate_taskdata(self):
        """Validate TaskData attributes."""
        assert len(set(self.param_names)) == len(self.param_names), "Duplicate parameter names"

        assert self.delta >= 0, f"Delta={self.delta} should be >= 0"

        assert self.eta >= 0, f"Eta={self.eta} should be >= 0"

        assert self.eta_decay_rate >= 0, f"Eta decay rate={self.eta_decay_rate} should be >= 0"

        assert (
            self.feedback in TaskData._FEEDBACK_VALUES
        ), f"Feedback={self.feedback} should be one of {TaskData._FEEDBACK_VALUES}."

        assert self.optimizer in self._OPTIMIZERS, f"optimizer={self.optimizer} should be one of {self._OPTIMIZERS}"
        if self.optimizer == "sgd":
            self.optimizer = SGD(**self.optimizer_kwargs)
        else:
            self.optimizer = RMSprop(**self.optimizer_kwargs)


class SessionData:
    """A helper class that holds data representing the session.

    Attributes:
        first_explore: The first point to explore as part of this session.
        second_explore: The second point to explore as part of this session.
        center: The point around which exploration is done.
        explore_direction: Direction along which exploration is done.
        first_explore_id: Identifier for first_explore.
        second_explore_id: Identifier for second_explore.
        first_reward: Used to store the reward from the first explore during two-point feedback.
    """

    @log_args_and_return(logger=_logger)
    def __init__(
        self,
        center: np.ndarray,
        explore_direction: np.ndarray,
        taskdata: TaskData,
        first_id: int,
    ):
        # Explore in the direction of explore_direction
        self.first_explore = BlueFin.compute_explore_value(center, 1, explore_direction, taskdata)
        # Explore in the direction opposite to explore_direction
        self.second_explore = (
            None
            if taskdata.feedback == "onepoint"
            else BlueFin.compute_explore_value(center, -1, explore_direction, taskdata)
        )
        self.center = center
        self.explore_direction = explore_direction
        self.first_explore_id = first_id
        self.second_explore_id = 0 if taskdata.feedback == "onepoint" else first_id + 1
        self.first_reward = None


class BlueFin:
    """BlueFin Class.

    Attributes:
        parameters: The parameters to tune.
        feedback: Type of feedback update. Can take values {'onepoint', 'twopoint'}.
        eta: The learning rate.
        delta: The exploration radius.
        optimizer: The optimizer to use. Can take values {"sgd", "rmsprop"}.
        optimizer_kwargs: The optimizer keyword arguments. For RMSProp, it can take values for
          {"alpha", "momentum", "eps"}. SGD does not accept any keyword arguments.
        random_seed: The random seed used to initialize the numpy pseudo-random number generator.
        eta_decay_rate: The decay rate of eta. Defaults to 0 where eta does not decay after each round.
        normalize: Specifies whether the parameter values have to be normalized. Uses min-max normalization.
        taskdata: Stores data relating to the learning task.
        random_state: The random number generator used by the algorithm.
    """

    _LOGGING_LEVEL_MAP = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
        "silent": logging.CRITICAL + 10,
    }

    @log_args_and_return(logger=_logger)
    def __init__(
        self,
        parameters: Iterable[Union[ContinuousParameter, DiscreteParameter]],
        feedback: str = "onepoint",
        eta: float = 0.01,
        delta: float = 0.1,
        optimizer: str = None,
        optimizer_kwargs: dict = None,
        random_seed: int = 123,
        eta_decay_rate: float = 0,
        normalize: bool = True,
        logging_level: str = "debug",
    ):
        _logger.setLevel(BlueFin._LOGGING_LEVEL_MAP[logging_level.lower()])
        _logger.debug(
            "New bluefin instance: parameters: %s, feedback: %s, eta: %s, delta: %s, optimizer: %s, "
            "optimizer_kwargs: %s, random_seed: %s, eta_decay_rate: %s, noramlize: %s, logging_level: %s",
            parameters,
            feedback,
            eta,
            delta,
            optimizer,
            optimizer_kwargs,
            random_seed,
            eta_decay_rate,
            normalize,
            logging_level,
        )

        self.round = 0
        self.seed = random_seed
        self.random_state = np.random.RandomState(seed=self.seed)

        param_names = [p.name for p in parameters]
        initial_values = np.asarray([p.initial_value for p in parameters], dtype=float)
        constraints = [p.constraint for p in parameters]

        if optimizer is None:
            optimizer = "sgd"

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        for idx, c in enumerate(constraints):
            if c.step_size is not None:
                # TODO Corner case: 1.7976931348623158E+308 / 0.1
                c.lb /= c.step_size
                c.ub /= c.step_size
                initial_values[idx] /= c.step_size

        self.taskdata = TaskData(
            param_names,
            initial_values,
            constraints,
            "bluefin",
            feedback,
            eta,
            delta,
            optimizer,
            optimizer_kwargs,
            eta_decay_rate,
            normalize,
        )

        if self.taskdata.normalize:
            initial_values = min_max_norm(initial_values, self.taskdata.lbs, self.taskdata.ubs)

        lr = BlueFin.get_lr(eta=self.taskdata.eta, delta=self.taskdata.delta)
        if lr != 0:
            initial_values = BlueFin.clip_center(initial_values, self.taskdata)

        self.session = SessionData(
            initial_values,
            self.sample_unit_sphere(initial_values.shape),
            self.taskdata,
            0,
        )

    @log_args_and_return(logger=_logger)
    def predict(self) -> Dict[str, Union[float, int]]:
        """
        Gets the next set of parameter values to use as a dictionary of key-value pairs.

        Returns:
            A dict of {parameter name : parameter value} key-value pairs.
        """
        param_dict = {}
        predict_values = (
            self.session.first_explore if self.session.first_reward is None else self.session.second_explore
        )

        converter = (float, round)
        for idx, param_name in enumerate(self.taskdata.param_names):
            param_dict[param_name] = converter[self.taskdata.is_ints[idx]](predict_values[idx])

        _logger.debug("Predicted values: %s", param_dict)
        return param_dict

    @log_args_and_return(logger=_logger)
    def set_reward(self, reward: float):
        """
        Sets the reward based on the parameters returned from predict.

        Args:
            reward: A measure of how well the parameter values returned in predict performed.
        """
        _logger.debug("Reward: %s", reward)
        if self.taskdata.feedback_denom == 1:  # onepoint
            new_center = BlueFin.compute_new_center(
                self.session.center,
                self.session.explore_direction,
                reward,
                self.taskdata,
            )
            self.session = SessionData(
                new_center,
                self.sample_unit_sphere(new_center.shape),
                self.taskdata,
                self.session.first_explore_id + 1,
            )
            self.round += 1
            self.eta_decay()

        else:  # twopoint
            if self.session.first_reward is None:
                self.session.first_reward = reward
            else:
                reward_diff = self.session.first_reward - reward

                # Normalizing reward diff
                reward_denominator = max(abs(self.session.first_reward), abs(reward))
                if reward_denominator != 0:
                    reward_diff /= reward_denominator

                new_center = BlueFin.compute_new_center(
                    self.session.center,
                    self.session.explore_direction,
                    reward_diff,
                    self.taskdata,
                )
                self.session = SessionData(
                    new_center,
                    self.sample_unit_sphere(new_center.shape),
                    self.taskdata,
                    self.session.second_explore_id + 1,
                )
                self.round += 1
                self.eta_decay()

    @log_args_and_return(logger=_logger)
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

        return sample

    @log_args_and_return(logger=_logger)
    def eta_decay(self):
        """Decay eta based on the decay rate and current round."""
        self.taskdata.eta = self.taskdata.initial_eta / (1 + self.taskdata.eta_decay_rate * self.round)
        self.taskdata.optimizer.eta = self.taskdata.eta
        self.taskdata.delta = np.sqrt(self.taskdata.eta)
        _logger.debug(
            "eta decay: eta: %s delta: %s eta_decay_rate: %s, round: %s",
            self.taskdata.eta,
            self.taskdata.delta,
            self.taskdata.eta_decay_rate,
            self.round,
        )

    @property
    def feedback(self) -> str:
        return self.taskdata.feedback

    @property
    def eta(self) -> float:
        return self.taskdata.eta

    @eta.setter
    def eta(self, e):
        assert e >= 0, f"Eta={e} should be >= 0"
        self.taskdata.eta = e

    @property
    def delta(self) -> float:
        return self.taskdata.delta

    @delta.setter
    def delta(self, d):
        assert d >= 0, f"Delta={d} should be >= 0"
        self.taskdata.delta = d

    @property
    def random_seed(self) -> int:
        return self.seed

    @property
    def eta_decay_rate(self) -> float:
        return self.taskdata.eta_decay_rate

    @eta_decay_rate.setter
    def eta_decay_rate(self, edr):
        assert edr >= 0, f"Eta decay rate={edr} should be >= 0"
        self.taskdata.eta_decay_rate = edr

    @property
    def normalize(self) -> bool:
        return self.taskdata.normalize

    @property
    def center(self) -> np.ndarray:
        return self.session.center

    @property
    def predict_id(self) -> int:
        return self.session.first_explore_id if self.session.first_reward is None else self.session.second_explore_id

    @staticmethod
    @log_args_and_return(logger=_logger)
    def clip_center(center: np.ndarray, taskdata: TaskData) -> np.ndarray:
        """
        Clips the center to the range [lb+delta, ub-delta].

        Args:
            center: The current parameter vector.
            taskdata: Data related to the learning task.

        Returns:
            The clipped center vector.
        """
        if taskdata.normalize is True:
            c_lb = Constraint.NORMALIZED_LB + taskdata.delta
            c_ub = Constraint.NORMALIZED_UB - taskdata.delta
        else:
            c_lb = taskdata.lbs + taskdata.delta
            c_ub = taskdata.ubs - taskdata.delta

        clipped_center = np.clip(a=center, a_min=c_lb, a_max=c_ub)

        return clipped_center

    @staticmethod
    @log_args_and_return(logger=_logger)
    def get_lr(eta: float, delta: float) -> float:
        lr = 0  # Return 0 if delta is 0

        if delta != 0:
            lr = eta / delta

        return lr

    @staticmethod
    @log_args_and_return(logger=_logger)
    def compute_new_center(
        center: np.ndarray,
        explore_direction: np.ndarray,
        reward_grad: float,
        taskdata: TaskData,
    ) -> np.ndarray:
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
        lr = BlueFin.get_lr(eta=taskdata.eta, delta=taskdata.delta)

        if lr != 0:  # Delta = 0
            grad = center.size * reward_grad * explore_direction / (taskdata.feedback_denom * taskdata.delta)
            step_value = taskdata.optimizer.get_step_value(grad)
            new_center = center + step_value
        else:
            step_value = 0
            new_center = center

        new_center = BlueFin.clip_center(new_center, taskdata)

        _logger.debug(
            "old center: %s, center size: %s, reward_grad: %s, explore_direction: %s, feedback_denom: %s, delta: %s, eta: %s, step_value: %s",
            center,
            center.size,
            reward_grad,
            explore_direction,
            taskdata.feedback_denom,
            taskdata.delta,
            taskdata.eta,
            step_value
        )
        _logger.debug("new center: %s", new_center)
        return new_center

    @staticmethod
    @log_args_and_return(logger=_logger)
    def compute_explore_value(
        center: np.ndarray,
        explore_sign: int,
        explore_direction: np.ndarray,
        taskdata: TaskData,
    ) -> np.ndarray:
        """Compute the point to explore.

        Computes the point to explore by randomly perturbing the center
        vector. This explore vector is sampled from a hypersphere centered at
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
        explore_value = center + change

        _logger.debug(
            "center: %s, explore_sign: %s, delta: %s, explore_direction: %s",
            center,
            explore_sign,
            taskdata.delta,
            explore_direction,
        )
        _logger.debug("Explore value before denormalization: %s", explore_value)

        if taskdata.normalize is True:
            explore_value = min_max_denorm(explore_value, taskdata.lbs, taskdata.ubs)

        _logger.debug("Explore value before project: %s", explore_value)
        explore_value = BlueFin.project(explore_value, taskdata.constraints)
        _logger.debug("Explore value after project: %s", explore_value)

        return explore_value

    @staticmethod
    @log_args_and_return(logger=_logger)
    def project(center: np.ndarray, constraints: Sequence[Constraint]) -> np.ndarray:
        """Projects vector based on constraints.

        Args:
            center: The vector of parameters to project.
            constraints: Constraints on the parameters.

        Returns:
            The projected vector.
        """
        new_center = []
        for idx, val in enumerate(center):
            new_center.append(BlueFin.project_single(val, constraints[idx]))

        return np.array(new_center)

    @staticmethod
    @log_args_and_return(logger=_logger)
    def project_single(val: float, constraint: Constraint) -> float:
        """
        Projects a single value based on constraints.

        Args:
            val: The value to project
            constraint: Constraints on the value

        Returns:
            The projected value.
        """
        is_int = constraint.is_int
        c_lb = constraint.lb
        c_ub = constraint.ub
        step_size = constraint.step_size

        new_val = min(max(c_lb, val), c_ub)

        if is_int and step_size is None:
            new_val = np.round(new_val, decimals=0)

        if step_size is not None:
            # This is an Arithmetic progression of the form
            # lb, lb+1, lb+2, ... ub
            # Assuming new_val = a + (n-1)*d, where d = 1, a = lb.
            # Solving for n, we get an approximate value which can be rounded off
            # to the nearest integer.
            n = np.round(new_val - constraint.lb + 1, decimals=0)
            # Keeping the (n - 1) in brackets helps prevent floating point imprecision
            new_val = (constraint.lb + (n - 1)) * step_size

        return new_val

    @staticmethod
    @log_args_and_return(logger=_logger)
    def get_normalized_constraints(num_params: int) -> Tuple[Constraint]:
        return tuple(
            Constraint(lb=Constraint.NORMALIZED_LB, ub=Constraint.NORMALIZED_UB, dtype="float")
            for _ in range(num_params)
        )

    def __str__(self):
        return "BlueFin"
