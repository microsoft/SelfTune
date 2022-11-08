import logging
from typing import Union

_logger = logging.getLogger("parameters.constraint")

__all__ = ("Constraint",)


class Constraint:
    """A class to hold constraints on a parameter learned using BlueFin.

    Attributes:
        lb: The minimum value a parameter can take.
        ub: The maximum value a parameter can take.
        dtype: The type of the parameter. Values can be ['float', 'int'].
        step_size: Unit of increment / decrement of the parameter. Defaults to None which assumes no step size.
        is_int: Specifies whether a parameter is an integer.
    """

    _DEFAULT_LB = -1.7976931348623157e308  # 64 bit floating point min value
    _DEFAULT_UB = 1.7976931348623158e308  # 64 bit floating point max value
    NORMALIZED_LB = 0.0
    NORMALIZED_UB = 1.0
    _DTYPES = (
        "float",
        "int",
    )

    def __init__(
        self,
        lb: Union[float, int] = None,
        ub: Union[float, int] = None,
        dtype: str = None,
        step_size: Union[float, int] = None,
    ):
        assert dtype in Constraint._DTYPES, f"Unknown dtype={dtype}, should be one of {Constraint._DTYPES}"
        self.lb = Constraint._DEFAULT_LB if lb is None else lb
        self.ub = Constraint._DEFAULT_UB if ub is None else ub
        assert self.lb <= self.ub, f"lb={self.lb} must be <= ub={self.ub}"
        if self.lb == self.ub:
            _logger.warning(
                f"Ideally, lb={self.lb} should be strictly less than ub={self.ub}."
                f" Since both bounds are equal, consider making this variable a constant."
            )

        self.dtype = dtype
        self.step_size = step_size

        if self.dtype:
            is_int = self.dtype == "int"
            if self.lb and not (isinstance(self.lb, int) == is_int):
                _logger.warning('lb=%s is not of type "%s"', self.lb, self.dtype)

            if self.ub and not (isinstance(self.ub, int) == is_int):
                _logger.warning('ub=%s is not of type "%s"', self.ub, self.dtype)

            if self.step_size and not (isinstance(self.step_size, int) == is_int):
                _logger.warning('step_size=%s is not of type "%s"', self.step_size, self.dtype)

        self.is_int = True if dtype == "int" else False

    def validate(
        self,
        value: Union[float, int],
        strict: bool = False,  # TODO Prefer setting it to True
        name: str = "?",
    ):
        """Validate Parameter attributes."""

        if strict:
            # For strict check:
            # 1. A float value, even with the decimal digits as 0, is not considered a valid integer.
            #    For example, 1.0 and 1.2 are not valid if the dtype is 'int'.
            #
            # 2. An integer is not considered a valid float.
            #    For example, 1 is not valid if the dtype is 'float'.
            #
            # To treat the above cases as valid, use strict=False.
            assert isinstance(value, int) == (self.dtype == "int"), f'[{name}] {value} is not of type "{self.dtype}"'
        else:
            if self.dtype == "int":
                assert (
                    isinstance(value, int) or value.is_integer()
                ), f'[{name}] {value} must be integer-like for dtype "int"'

        assert value >= self.lb, f"[{name}] value={value} must be >= lb={self.lb}"
        assert value <= self.ub, f"[{name}] value={value} must be <= ub={self.ub}"

        if self.step_size is not None:
            assert ((self.lb - value) / self.step_size).is_integer(), (
                f"[{name}] Invalid lower bound: lb={self.lb} is not reachable"
                f" using the given value={value} and step_size={self.step_size}"
            )
            assert ((self.ub - value) / self.step_size).is_integer(), (
                f"[{name}] Invalid upper bound for: ub={self.ub} is not reachable"
                f" using the given value={value} and step_size={self.step_size}"
            )

    def to_dict(self) -> dict:
        return {
            "lb": self.lb,
            "ub": self.ub,
            "dtype": self.dtype,
            "step_size": self.step_size,
        }

    @staticmethod
    def from_dict(data: dict) -> "Constraint":
        # Can also simply do: Constraint(**data)
        return Constraint(
            lb=data.get("lb"),
            ub=data.get("ub"),
            dtype=data.get("dtype"),
            step_size=data.get("step_size"),
        )

    def __repr__(self):
        return f"Constraint(lb={self.lb}, ub={self.ub}, dtype={self.dtype}, step_size={self.step_size})"
