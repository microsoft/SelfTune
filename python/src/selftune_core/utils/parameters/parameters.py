from typing import Hashable, Iterable, Union

from .constraint import Constraint

__all__ = (
    "Parameter",
    "ContinuousParameter",
    "DiscreteParameter",
    "OrdinalParameter",
    "CategoricalParameter",
    "create_parameter_from_dict",
)


class Parameter:
    pass


class ContinuousParameter(Parameter):
    def __init__(
        self,
        name: str,
        initial_value: float,
        constraint: Constraint = None,
        type: str = None,  # Not needed, only for easy deserialization
    ):
        if not constraint:
            constraint = Constraint(dtype="float")
        else:
            assert constraint.dtype == "float"

        constraint.validate(initial_value, name=name)

        self.name = name
        self.initial_value = initial_value
        self.constraint = constraint

    def to_dict(self) -> dict:
        return {
            "type": "continuous",
            "name": self.name,
            "initial_value": self.initial_value,
            **self.constraint.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "ContinuousParameter":
        return ContinuousParameter(
            name=data["name"],
            initial_value=data["initial_value"],
            constraint=Constraint(
                lb=data.get("lb"),
                ub=data.get("ub"),
                dtype="float",
                step_size=data.get("step_size"),
            ),
        )

    def __repr__(self):
        return "ContinuousParameter(name={}, initial_value={}, constraint={})".format(
            self.name, self.initial_value, self.constraint
        )


class DiscreteParameter(Parameter):
    def __init__(
        self,
        name: str,
        initial_value: int,
        constraint: Constraint = None,
        type: str = None,  # Not needed, only for easy deserialization
    ):
        if not constraint:
            constraint = Constraint(dtype="int", step_size=1)
        else:
            assert constraint.dtype == "int"
            if constraint.step_size is None:
                constraint.step_size = 1

        constraint.validate(initial_value, name=name)

        self.name = name
        self.initial_value = initial_value
        self.constraint = constraint

    def to_dict(self) -> dict:
        return {
            "type": "discrete",
            "name": self.name,
            "initial_value": self.initial_value,
            **self.constraint.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "DiscreteParameter":
        return DiscreteParameter(
            name=data["name"],
            initial_value=data["initial_value"],
            constraint=Constraint(
                lb=data.get("lb"),
                ub=data.get("ub"),
                dtype="int",
                step_size=data.get("step_size", 1),
            ),
        )

    def __repr__(self):
        return "DiscreteParameter(name={}, initial_value={}, constraint={})".format(
            self.name, self.initial_value, self.constraint
        )


class OrdinalParameter(Parameter):
    def __init__(
        self,
        name: str,
        initial_value: Hashable,
        values: Iterable[Hashable],
        type: str = None,  # Not needed, only for easy deserialization
    ):
        self.name = name
        self.initial_value = initial_value
        self.values = tuple(values)
        self.__map = {value: i for i, value in enumerate(values)}
        self.constraint = Constraint(lb=0, ub=len(self.values) - 1, dtype="int", step_size=1)

    def value(self, mapped_value: int):
        """mapped_value is the index in this case"""
        return self.values[mapped_value]

    def mapped_value(self, value) -> int:
        return self.__map[value]

    def to_dict(self) -> dict:
        return {
            "type": "ordinal",
            "name": self.name,
            "initial_value": self.initial_value,
            "values": self.values,
            **self.constraint.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "OrdinalParameter":
        return OrdinalParameter(
            name=data["name"],
            initial_value=data["initial_value"],
            values=data["values"],
        )

    def __getitem__(self, mapped_value: int):
        """mapped_value is the index in this case"""
        return self.values[mapped_value]

    def __repr__(self):
        return "OrdinalParameter(name={}, initial_value={}, values={})".format(
            self.name, self.initial_value, self.values
        )


class CategoricalParameter(Parameter):
    def __init__(
        self,
        name: str,
        initial_value: Hashable,
        values: Iterable[Hashable],
        type: str = None,  # Not needed, only for easy deserialization
    ):
        self.name = name
        self.initial_value = initial_value
        self.values = tuple(values)  # TODO One-hot encoding

    def to_dict(self) -> dict:
        return {
            "type": "categorical",
            "name": self.name,
            "initial_value": self.initial_value,
            "values": self.values,
        }

    @staticmethod
    def from_dict(data: dict) -> "CategoricalParameter":
        return CategoricalParameter(
            name=data["name"],
            initial_value=data["initial_value"],
            values=data["values"],
        )

    def __repr__(self):
        return "CategoricalParameter(name={}, initial_value={}, values={})".format(
            self.name, self.initial_value, self.values
        )


_PARAMETER_CLASS = {
    "continuous": ContinuousParameter,
    "discrete": DiscreteParameter,
    "ordinal": OrdinalParameter,
    "categorical": CategoricalParameter,
}


def create_parameter_from_dict(
    data: dict,
) -> Union[ContinuousParameter, DiscreteParameter, OrdinalParameter, CategoricalParameter]:
    return _PARAMETER_CLASS[data["type"]].from_dict(data)
