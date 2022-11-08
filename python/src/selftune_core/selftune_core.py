from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Union

from .backends.bluefin import BlueFin
from .utils.parameters import Parameter, create_parameter_from_dict

__all__ = ("SelfTune",)

_BACKENDS = {
    "bluefin": BlueFin,
}

_DEFAULT_ALGORITHM = "bluefin"
_DEFAULT_ALGORITHM_CONFIG = dict(
    feedback="onepoint",
    eta=0.01,
    delta=0.1,
    normalize=True,
    random_seed=123,
)


class SelfTune:
    def __init__(
        self,
        parameters: Iterable[Union[dict, Parameter]],
        algorithm: Optional[str] = None,
        algorithm_args: Optional[dict] = None,
    ):
        """
        Finds the correct backend corresponding to the :code:`algorithm`, and initializes it using
        the :code:`parameters` and :code:`algorithm_args`.

        Args:
            parameters: The list of parameters to tune
            algorithm: The algorithm SelfTune should use
            algorithm_args: The arguments required by the chosen algorithm
        """
        if algorithm is None:
            assert algorithm_args is None, "Must specify the algorithm with the algorithm arguments"
            self.algorithm = _DEFAULT_ALGORITHM
            algorithm_args = _DEFAULT_ALGORITHM_CONFIG
        else:
            self.algorithm = algorithm
            algorithm_args = algorithm_args or {}

        parameters = tuple((create_parameter_from_dict(p) if isinstance(p, dict) else deepcopy(p) for p in parameters))
        self.backend = _BACKENDS[self.algorithm](parameters=parameters, **algorithm_args)

    def predict(self, **kwargs) -> Dict[str, Union[bool, float, int, str]]:
        """
        Calls the backend's predict method

        Args:
            **kwargs: Any keyword arguments required by the backend

        Returns:
            A dictionary, having keys as the parameter names and the values as the parameter values.
        """
        return self.backend.predict(**kwargs)

    def set_reward(self, reward: Union[float, int], **kwargs):
        """
        Calls the backend's set_reward method

        Args:
            reward: The reward needed by the backend to update
            **kwargs: Any keyword arguments required by the backend

        Returns:
            None
        """
        self.backend.set_reward(reward, **kwargs)

    def __getattr__(self, attr) -> Any:
        """
        This function allows the user to access backend specific fields.
        Let us take the example of BlueFin as the backend. If the user was
        using BlueFin directly, then they could access backend fields as follows:

        .. code-block:: python
            backend = BlueFin()
            backend.eta

        Now, using SelfTune as a wrapper over BlueFin, the user can access
        the field as

        .. code-block:: python
            st = SelfTune(algorithm='bluefin')
            st.eta
            # The above line invokes SelfTune.__getattr__('eta')
            # Which evaluates to: return getattr(self.backend, 'eta')
            # which means: return self.backend.eta
            # Note that here self.backend is an instance of BlueFin

        Important:
            Fields that exist in SelfTune are not affected. So, for example,
            :code:`st.algorithm` will still refer to the algorithm field in
            SelfTune and not in self.backend
        """
        return getattr(self.backend, attr)

    def __str__(self):
        return f'SelfTune(algorithm="{self.algorithm}")'
