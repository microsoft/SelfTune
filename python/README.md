## Installation

To setup the package locally, run

```bash
$ pip install .
```

## Usage

Refer to `examples/bluefin` and `tests/test_bluefin.py` for examples.

```python
import numpy as np

from selftune_core import SelfTune


def get_reward(pred) -> float:
    """Negative squared loss."""
    target = np.array([1, 700])
    return -np.square(pred - target).sum() / (1000**2)


def main():
    parameters = (
        {
            "type": "discrete",
            "name": "p1",
            "initial_value": 5,
            "lb": 0,
            "ub": 10,
        },
        {
            "type": "continuous",
            "name": "p2",
            "initial_value": 100.0,
            "lb": 100.0,
            "ub": 900.0,
            "step_size": 100.0,
        },
    )

    # Initialize an instance of SelfTune
    st = SelfTune(
        algorithm="bluefin",
        parameters=parameters,
        algorithm_args=dict(feedback="twopoint", eta=0.01, delta=0.1, random_seed=4),
    )

    num_iterations = 100

    for i in range(num_iterations):
        # Predict the next set of perturbed parameters
        pred = st.predict()

        # Receive feedback
        reward = get_reward(np.asarray([pred["p1"], pred["p2"]]))

        # Send the feedback to SelfTune for the gradient update
        st.set_reward(reward)

        if i % 25 == 0:
            print(
                f'Round={i}, Reward={reward}, Pred=({pred["p1"]:4f}, {pred["p2"]:.4f}),'
                f" Best=({st.center[0]:.4f}, {st.center[1]:.4f})"
            )

if __name__ == '__main__':
    main()
```