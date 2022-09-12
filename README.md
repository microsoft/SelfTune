<p align="center">
<img src=".github/selftune_banner.png" width=500 alt="SelfTune: An RL framework to tune configuration parameters">
</p>

<p align="center">
<a href="https://github.com/microsoft/SelfTune/blob/main/LICENSE"><strong>License</strong></a> •
<a href="https://github.com/microsoft/SelfTune/blob/main/SECURITY.md"><strong>Security</strong></a> •
<a href="https://github.com/microsoft/SelfTune/blob/main/SUPPORT.md"><strong>Support</strong></a> •
<a href="https://github.com/microsoft/SelfTune/blob/main/CODE_OF_CONDUCT.md"><strong>Code of Conduct</strong></a>
</p>

# SelfTune

SelfTune is an RL framework that enables systems and service developers to automatically tune various configuration parameters and other heuristics in their codebase, rather than manually-tweaking, over time in deployment. It provides easy-to-use API (Python, C# bindings) and is driven by bandit-style RL & online gradient-descent algorithms.

## Installation and Usage

Refer to the [python README](python/README.md) and [C# README](c%23/README.md).

## Basic tour of the SelfTune package
In this section, we present the syntax and semantics of SelfTune's python bindings. These ideas apply to the C# bindings too.

### 1. Identifying the reward function

SelfTune's optimization algorithm(e.g., Bluefin) uses a reward to compute a gradient-ascent style update to the parameter values. This reward can be any health or utiliziation metric of the current state of the system (e.g., throughput, latency, ...).

### 2. Defining the parameters to be tuned
We define the parameters to be tuned. The library allows optional arguments that encode domain knowledge for tuning the parameters:
<ol type="a">
    <li>The initial value of the parameter</li>
    <li>(optional) Constraints on the parameter to be tuned. We currently support range constraints (c_min and c_max), type constraints (is_int = True if the parameter takes only integral values)</li>
</ol>

```python
import numpy as np
from selftune import  SelfTune, Constraints

initial_values = np.array([0.9, 0.4])
constraints = [
    Constraints(c_min=0, c_max=1, is_int=False),
    Constraints() # No constraints on second parameter
]
```

### 3. Create an instance of SelfTune
Once we define the parameters to be tuned, we can create an instance of the parameter learning problem for SelfTune. A description of the model hyperparameters and the best practices when defining these values is defined below.

- <strong>opt</strong> - The optimization algorithm to use. Currently, we only support the `bluefin` algorithm.
- <strong>feedback</strong> - The type of feedback update. The feedback can be either `onepoint` or `twopoint`. `onepoint` is recommended when the reward function changes with time i.e., it is not possible to query the reward function at the same set of parameters twice and expect the same reward. In settings (e.g., simulations) where it is possible to obtain the reward at two different set of parameters, `twopoint` is prefered since it is more sample-efficient and converges faster.

Along with the above arguments, the user can also optionally provide
- <strong>eta</strong> - The learning rate. 
- <strong>delta</strong> - The exploration radius.
- <strong>random_state</strong> - The random seed used to initialize the numpy pseudo-random number generator in the library.
- <strong>eta_decay_rate</strong> - The decay rate of eta.


```python
model = SelfTune(initial_values=initial_values,
                 constraints=constraints,
                 opt='bluefin',
                 feedback='onepoint')
```

The user can also modify the eta after each round. For example,
```python
def eta_decay(inital_eta, curr_round, decay_rate):
    return (decay_rate**curr_round)*initial_eta

num_rounds = 100
decay_rate = 0.95
initial_eta = 0.01

model = SelfTune(initial_values=initial_values,
                 constraints=constraints,
                 opt='bluefin',
                 feedback='onepoint')

for round in range(1, num_rounds):
    model.eta = eta_decay(initial_eta, round, decay_rate)
    model.delta = model.eta**0.5
```

In cases where both `onepoint` and `twopoint` feedback are applicable, it is recommended to use `twopoint`. `twopoint` provides more stable convergence and more accurate gradient estimates compared to `onepoint`.

<table>
    <tr>
        <td>
            <p align="center">
                <img src=".github/onepoint_twopoint_2parameter.png" alt="Onepoint vs Twopoint loss decay for a 2 parameter learning problem" width=700/>
            </p>
        </td>
        <td>
            <p align="center">
                <img src=".github/onepoint_twopoint_5parameter.png" alt="Onepoint vs Twopoint loss decay for a 5 parameter learning problem" width=700/></td>
            </p>
    </tr>
</table>

### 4. Predict, Set Reward
Now that we have set up an instance of SelfTune, we can call `model.predict` to get the current set of parameters. Once the reward is available, it can be sent back to SelfTune using `model.set_reward`. In the plots comparing `onepoint` and `twopoint` feedback, we use the negative squared loss as the reward function.

```python
num_rounds = 100
for i in range(num_rounds):
    # Get the current set of parameters
    pred = model.predict()

    # Receive feedback
    reward = black_box_reward(pred)

    # Send the feedback to SelfTune for the gradient update
    st.set_reward(reward)
    
    if i % 5 == 0:
        print(
            f'Round={i}, Reward={reward}, Pred=({pred[0]:.4f}, {pred[1]:.4f}), Best=({st.center[0]}, {st.center[1]})'
        )
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
