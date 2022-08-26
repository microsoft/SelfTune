# SelfTune

SelfTune is a framework that uses data-driven tuning of parameters that produce larger rewards.

## Usage

```python
import numpy as np
from selftune import SelfTune, Constraints


def get_reward(pred):
    target = np.array([0.1, 0.6])
    return -(np.linalg.norm(pred - target)**2)


def main():
    initial_values = np.array([0.9, 0.5])
    constraints = [
        Constraints(c_min=0, c_max=1, is_int=False),
        Constraints(c_min=0, c_max=1, is_int=False)
    ]

    # Initialize an instance of SelfTune
    st = SelfTune(initial_values=initial_values,
                  constraints=constraints,
                  opt='bluefin',
                  feedback='onepoint',
                  eta=0.01,
                  delta=0.1)

    num_iterations = 100
    for i in range(num_iterations):
        # Predict the next set of perturbed parameters
        pred = st.predict()

        # Receive feedback
        reward = get_reward(pred)

        # Send the feedback to SelfTune for the gradient update
        st.set_reward(reward)
        
        if i % 5 == 0:
            print(
                f'Round={i}, Reward={reward}, Pred=({pred[0]:.4f}, {pred[1]:.4f}), Best=({st.center[0]}, {st.center[1]})'
            )


if __name__ == '__main__':
    main()
```