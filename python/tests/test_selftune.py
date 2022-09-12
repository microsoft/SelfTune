import unittest
import numpy as np
from selftune import SelfTune, Constraints


class TestSelfTune(unittest.TestCase):
    def test_consecutive_predict(self):
        """Consecutive predict calls should return the same values."""
        initial_values = np.array([4, 4])
        constraints = [
            Constraints(c_min=1, c_max=10, is_int=False),
            Constraints(c_min=1, c_max=10, is_int=False)
        ]

        st = SelfTune(initial_values=initial_values, constraints=constraints)

        pred1 = st.predict()
        pred2 = st.predict()

        self.assertTrue((pred1 == pred2).all())

    def test_constraint(self):
        """Test for min larger than max in constraint."""
        self.assertRaises(AssertionError, Constraints, 10, 1, False)

    def test_onepoint(self):
        """Testing onepoint."""
        def get_reward(prediction: np.ndarray):
            """Squared Loss."""
            target = np.array([1, 7])
            return -((prediction[0] - target[0]) *
                     (prediction[0] - target[0]) +
                     (prediction[1] - target[1]) *
                     (prediction[1] - target[1])) / 10

        _EXPECTED_REWARDS = [
            -4.505402822180811, -6.093862535845089, -8.633547180765486,
            -7.1131380599201055, -0.46653811179493143
        ]

        initial_values = np.array([5, 2])
        constraints = [
            Constraints(c_min=1, c_max=10, is_int=False),
            Constraints(c_min=1, c_max=10, is_int=True)
        ]

        st = SelfTune(initial_values=initial_values,
                      constraints=constraints,
                      feedback='onepoint',
                      eta=1,
                      delta=1,
                      random_state=2)
        num_iterations = 5

        rewards = []
        for _ in range(num_iterations):
            pred = st.predict()

            reward = get_reward(pred)
            rewards.append(reward)

            st.set_reward(reward)

        self.assertListEqual(rewards, _EXPECTED_REWARDS)

    def test_twopoint(self):
        """Testing twopoint."""
        def get_reward(prediction: np.ndarray):
            """Absolute Loss."""
            target = np.array([2, 8])
            return -(np.abs(prediction[0] - target[0]) +
                     np.abs(prediction[1] - target[1])) / 10

        _EXPECTED_REWARDS = [
            -0.5142788156834177, -0.6857211843165822, -0.4383506903150892,
            -0.71878569917557, -0.47582499065196454
        ]

        initial_values = np.array([4, 4])
        constraints = [
            Constraints(c_min=1, c_max=10, is_int=False),
            Constraints(c_min=1, c_max=10, is_int=False)
        ]

        st = SelfTune(initial_values=initial_values,
                      constraints=constraints,
                      feedback='twopoint',
                      eta=1,
                      delta=1,
                      random_state=2)
        num_iterations = 5

        rewards = []
        for _ in range(num_iterations):
            pred = st.predict()

            reward = get_reward(pred)
            rewards.append(reward)

            st.set_reward(reward)

        self.assertListEqual(rewards, _EXPECTED_REWARDS)
