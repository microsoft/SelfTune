import numpy as np

from selftune_core import SelfTune


def get_reward(pred) -> float:
    """Negative RMSE Loss."""
    target = np.asarray([0.1, 0.5])
    return -np.sqrt(np.mean(np.square(pred - target)))


def main():
    parameters = (
        {
            "type": "continuous",
            "name": "p1",
            "initial_value": 0.8,
            "lb": 0.0,
            "ub": 1.0,
        },
        {
            "type": "continuous",
            "name": "p2",
            "initial_value": 0.3,
            "lb": 0.0,
            "ub": 1.0,
        },
    )

    # Initialize an instance of SelfTune
    st = SelfTune(
        algorithm="bluefin",
        parameters=parameters,
        algorithm_args=dict(
            feedback="twopoint",
            eta=0.01,
            delta=0.1,
            random_seed=4,
        ),
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


if __name__ == "__main__":
    main()
