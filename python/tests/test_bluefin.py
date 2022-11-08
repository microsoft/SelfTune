import unittest

import numpy as np

from selftune_core import Constraint, ContinuousParameter, DiscreteParameter, SelfTune


class TestSelfTune(unittest.TestCase):
    def test_getters(self):
        """Test for getters."""
        parameters = (
            ContinuousParameter(
                name="p1",
                initial_value=5.0,
                constraint=Constraint(
                    lb=1.0,
                    ub=10.0,
                    dtype="float",
                ),
            ),
        )

        st = SelfTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback="onepoint",
                eta=1,
                delta=1,
                random_seed=2,
                eta_decay_rate=0.1,
                normalize=False,
            ),
        )

        self.assertEqual(st.feedback, "onepoint")
        self.assertEqual(st.eta, 1)
        self.assertEqual(st.delta, 1)
        self.assertEqual(st.random_seed, 2)
        self.assertEqual(st.eta_decay_rate, 0.1)
        self.assertEqual(st.normalize, False)

    def test_duplicate_params(self):
        """Test for same parameter name."""

        parameters = (
            ContinuousParameter(
                name="p1",
                initial_value=0.9,
                constraint=Constraint(lb=0.0, ub=1.0, dtype="float"),
            ),
            ContinuousParameter(
                name="p1",
                initial_value=0.5,
                constraint=Constraint(lb=0.0, ub=1.0, dtype="float"),
            ),
        )

        self.assertRaises(AssertionError, SelfTune, parameters)

    def test_consecutive_predict(self):
        """Consecutive predict calls should return the same values."""
        parameters = (
            ContinuousParameter(
                name="p1",
                initial_value=4.0,
                constraint=Constraint(
                    lb=1.0,
                    ub=10.0,
                    dtype="float",
                ),
            ),
            ContinuousParameter(
                name="p2",
                initial_value=4.0,
                constraint=Constraint(
                    lb=1.0,
                    ub=10.0,
                    dtype="float",
                ),
            ),
        )

        st = SelfTune(parameters=parameters, algorithm="bluefin")

        pred1 = st.predict()
        pred2 = st.predict()

        self.assertDictEqual(pred1, pred2)

    def test_constraint(self):
        """Test assertions in Constraint."""

        # Initial value < lb
        self.assertRaises(
            AssertionError,
            ContinuousParameter,
            "p1",
            0.0,
            Constraint(
                lb=1.0,
                ub=10.0,
                dtype="float",
            ),
        )

        # Initial value > ub
        self.assertRaises(
            AssertionError,
            ContinuousParameter,
            "p2",
            11.0,
            Constraint(
                lb=1.0,
                ub=10.0,
                dtype="float",
            ),
        )

        # lb > ub
        self.assertRaises(AssertionError, Constraint, lb=10.0, ub=1.0, dtype="float")

        # lb not reachable with initial value
        self.assertRaises(
            AssertionError,
            DiscreteParameter,
            "p4",
            1,
            Constraint(
                lb=0,
                ub=9,
                step_size=2,
                dtype="int",
            ),
        )

        # ub not reachable with initial value
        self.assertRaises(
            AssertionError,
            DiscreteParameter,
            "p5",
            100,
            Constraint(
                lb=0,
                ub=750,
                step_size=100,
                dtype="int",
            ),
        )

    def test_onepoint(self):
        """Testing onepoint."""

        def get_reward(prediction: np.ndarray):
            """Squared Loss."""
            target = np.array(
                [
                    1,
                    7,
                ]
            )
            return -np.square(prediction - target).sum() / 100

        _EXPECTED_REWARDS = [
            -0.3405402822180811,
            -0.3106839096933405,
            -0.4316930647412465,
            -0.7599077620901906,
            -0.3736109295349623,
        ]

        parameters = (
            ContinuousParameter(
                name="p1",
                initial_value=5.0,
                constraint=Constraint(
                    lb=1.0,
                    ub=10.0,
                    dtype="float",
                ),
            ),
            DiscreteParameter(
                name="p2",
                initial_value=2,
                constraint=Constraint(
                    lb=1,
                    ub=10,
                    dtype="int",
                ),
            ),
        )

        st = SelfTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(feedback="onepoint", eta=1, delta=1, random_seed=2, normalize=False),
        )
        num_iterations = 5

        rewards = []
        for _ in range(num_iterations):
            pred = st.predict()

            reward = get_reward(
                np.asarray(
                    [
                        pred["p1"],
                        pred["p2"],
                    ]
                )
            )
            rewards.append(reward)

            st.set_reward(reward)

        self.assertListEqual(rewards, _EXPECTED_REWARDS)

    def test_onepoint_normalized(self):
        """Testing normalization with onepoint feedback."""

        def get_reward(prediction: np.ndarray):
            """Squared Loss."""
            target = np.array([10, 70])
            return -np.square(prediction - target).sum() / (100**2)

        _EXPECTED_REWARDS = [
            -0.3512376516560475,
            -0.3373101975411264,
            -0.5117596837324212,
            -0.7597980444916118,
            -0.4006154976867796,
        ]

        parameters = (
            ContinuousParameter(
                name="p1",
                initial_value=50.0,
                constraint=Constraint(
                    lb=1.0,
                    ub=100.0,
                    dtype="float",
                ),
            ),
            DiscreteParameter(
                name="p2",
                initial_value=20,
                constraint=Constraint(
                    lb=1,
                    ub=100,
                    dtype="int",
                ),
            ),
        )

        st = SelfTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback="onepoint",
                eta=0.01,
                delta=0.1,
                random_seed=2,
                normalize=True,
            ),
        )
        num_iterations = 5

        rewards = []
        for _ in range(num_iterations):
            pred = st.predict()

            reward = get_reward(
                np.asarray(
                    [
                        pred["p1"],
                        pred["p2"],
                    ]
                )
            )
            rewards.append(reward)

            st.set_reward(reward)

        self.assertListEqual(rewards, _EXPECTED_REWARDS)

    def test_twopoint_normalized(self):
        """Testing normalization with twopoint feedback."""

        def get_reward(prediction: np.ndarray):
            """Absolute Loss."""
            target = np.array([200, 800])
            return -np.abs(prediction - target).sum() / 1000

        _EXPECTED_REWARDS = [
            -0.5571822684338672,
            -0.6428177315661328,
            -0.5242572259680681,
            -0.6643345128938783,
            -0.5561020678470506,
        ]

        parameters = (
            ContinuousParameter(
                name="p1",
                initial_value=400.0,
                constraint=Constraint(
                    lb=1.0,
                    ub=1000.0,
                    dtype="float",
                ),
            ),
            ContinuousParameter(
                name="p2",
                initial_value=400.0,
                constraint=Constraint(
                    lb=1.0,
                    ub=1000.0,
                    dtype="float",
                ),
            ),
        )

        st = SelfTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback="twopoint",
                eta=0.0025,
                delta=0.05,
                optimizer="sgd",
                normalize=True,
                eta_decay_rate=0.03,
                random_seed=2,
            ),
        )
        num_iterations = 5

        rewards = []
        for _ in range(num_iterations):
            pred = st.predict()

            reward = get_reward(
                np.asarray(
                    [
                        pred["p1"],
                        pred["p2"],
                    ]
                )
            )
            rewards.append(reward)

            st.set_reward(reward)

        self.assertListEqual(rewards, _EXPECTED_REWARDS)

    def test_twopoint(self):
        """Testing twopoint."""

        def get_reward(prediction: np.ndarray):
            """Absolute Loss."""
            target = np.array([2, 8])
            return -np.abs(prediction - target).sum() / 10

        _EXPECTED_REWARDS = [
            -0.5,
            -0.7,
            -0.4,
            -0.7,
            -0.5,
        ]

        parameters = (
            DiscreteParameter(
                name="p1",
                initial_value=4,
                constraint=Constraint(
                    lb=1,
                    ub=10,
                    dtype="int",
                ),
            ),
            DiscreteParameter(
                name="p2",
                initial_value=4,
                constraint=Constraint(
                    lb=1,
                    ub=10,
                    dtype="int",
                ),
            ),
        )

        st = SelfTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback="twopoint",
                eta=0.01,
                delta=0.1,
                optimizer="sgd",
                random_seed=2,
            ),
        )
        num_iterations = 5

        rewards = []
        for _ in range(num_iterations):
            pred = st.predict()

            reward = get_reward(
                np.asarray(
                    [
                        pred["p1"],
                        pred["p2"],
                    ]
                )
            )
            rewards.append(reward)

            st.set_reward(reward)

        self.assertListEqual(rewards, _EXPECTED_REWARDS)

    def test_twopoint_zero_reward(self):
        """Test if zero division error occurs when reward is zero."""

        def get_reward():
            return 0

        parameters = (
            DiscreteParameter(
                name="p1",
                initial_value=4,
                constraint=Constraint(
                    lb=1,
                    ub=10,
                    dtype="int",
                ),
            ),
            DiscreteParameter(
                name="p2",
                initial_value=4,
                constraint=Constraint(
                    lb=1,
                    ub=10,
                    dtype="int",
                ),
            ),
        )

        st = SelfTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback="twopoint",
                eta=0.01,
                delta=0.1,
                optimizer="sgd",
                random_seed=2,
            ),
        )
        num_iterations = 5

        for _ in range(num_iterations):
            st.predict()

            reward = get_reward()

            st.set_reward(reward)

    def test_step_size(self):
        """Testing Step size."""

        def get_reward(prediction: np.ndarray):
            """Squared Loss."""
            target = np.array([1, 700])
            return -np.square(prediction - target).sum() / (1000**2)

        _EXPECTED_REWARDS = [
            -0.160016,
            -0.360016,
            -0.160009,
            -0.250025,
            -0.250016,
        ]

        parameters = (
            DiscreteParameter(
                name="p1",
                initial_value=5,
                constraint=Constraint(
                    lb=0,
                    ub=10,
                    dtype="int",
                ),
            ),
            ContinuousParameter(
                name="p2",
                initial_value=100.0,
                constraint=Constraint(
                    lb=100.0,
                    ub=900.0,
                    dtype="float",
                    step_size=100.0,
                ),
            ),
        )

        st = SelfTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback="twopoint",
                eta=0.01,
                delta=0.1,
                optimizer="sgd",
                random_seed=4,
            ),
        )
        num_iterations = 5

        rewards = []
        for _ in range(num_iterations):
            pred = st.predict()

            reward = get_reward(
                np.asarray(
                    [
                        pred["p1"],
                        pred["p2"],
                    ]
                )
            )
            rewards.append(reward)

            st.set_reward(reward)

        self.assertListEqual(rewards, _EXPECTED_REWARDS)

    def test_zero_eta(self):
        """Zero eta and delta should return the same parameters."""
        _EXPECTED_PARAM_VALUES_1 = [
            {"p1": 5, "p2": 100.0},
            {"p1": 5, "p2": 100.0},
            {"p1": 5, "p2": 100.0},
            {"p1": 5, "p2": 100.0},
            {"p1": 5, "p2": 100.0},
        ]

        parameters = (
            DiscreteParameter(
                name="p1",
                initial_value=5,
                constraint=Constraint(
                    lb=0,
                    ub=10,
                    dtype="int",
                ),
            ),
            ContinuousParameter(
                name="p2",
                initial_value=100.0,
                constraint=Constraint(
                    lb=100.0,
                    ub=900.0,
                    dtype="float",
                    step_size=100.0,
                ),
            ),
        )

        st = SelfTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback="twopoint",
                eta=0,
                delta=0,
                optimizer="sgd",
                random_seed=4,
            ),
        )
        num_iterations = 5

        param_values = []
        for _ in range(num_iterations):
            pred = st.predict()

            reward = -1  # Arbitrary reward
            param_values.append(pred)

            st.set_reward(reward)

        for idx, values in enumerate(param_values):
            self.assertDictEqual(values, _EXPECTED_PARAM_VALUES_1[idx])

        _EXPECTED_PARAM_VALUES_2 = [
            {
                "p1": 64,
                "p2": 40,
                "p3": 64,
                "p4": 40,
                "p5": 64,
                "p6": 40,
                "p7": 64,
                "p8": 40,
            },
            {
                "p1": 64,
                "p2": 40,
                "p3": 64,
                "p4": 40,
                "p5": 64,
                "p6": 40,
                "p7": 64,
                "p8": 40,
            },
            {
                "p1": 64,
                "p2": 40,
                "p3": 64,
                "p4": 40,
                "p5": 64,
                "p6": 40,
                "p7": 64,
                "p8": 40,
            },
            {
                "p1": 64,
                "p2": 40,
                "p3": 64,
                "p4": 40,
                "p5": 64,
                "p6": 40,
                "p7": 64,
                "p8": 40,
            },
            {
                "p1": 64,
                "p2": 40,
                "p3": 64,
                "p4": 40,
                "p5": 64,
                "p6": 40,
                "p7": 64,
                "p8": 40,
            },
        ]

        parameters = (
            DiscreteParameter(
                name="p1",
                initial_value=64,
                constraint=Constraint(
                    lb=64,
                    ub=19064,
                    step_size=1000,
                    dtype="int",
                ),
            ),
            DiscreteParameter(
                name="p2",
                initial_value=40,
                constraint=Constraint(
                    lb=10,
                    ub=40,
                    step_size=10,
                    dtype="int",
                ),
            ),
            DiscreteParameter(
                name="p3",
                initial_value=64,
                constraint=Constraint(
                    lb=64,
                    ub=100,
                    step_size=1,
                    dtype="int",
                ),
            ),
            DiscreteParameter(
                name="p4",
                initial_value=40,
                constraint=Constraint(
                    lb=1,
                    ub=40,
                    step_size=1,
                    dtype="int",
                ),
            ),
            ContinuousParameter(
                name="p5",
                initial_value=64.0,
                constraint=Constraint(
                    lb=64.0,
                    ub=19064.0,
                    step_size=1000.0,
                    dtype="float",
                ),
            ),
            ContinuousParameter(
                name="p6",
                initial_value=40.0,
                constraint=Constraint(
                    lb=10.0,
                    ub=40.0,
                    step_size=10.0,
                    dtype="float",
                ),
            ),
            ContinuousParameter(
                name="p7",
                initial_value=64.0,
                constraint=Constraint(
                    lb=64.0,
                    ub=19064.0,
                    step_size=1.0,
                    dtype="float",
                ),
            ),
            ContinuousParameter(
                name="p8",
                initial_value=40.0,
                constraint=Constraint(
                    lb=1.0,
                    ub=40.0,
                    step_size=1.0,
                    dtype="float",
                ),
            ),
        )

        st = SelfTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback="twopoint",
                eta=0,
                delta=0,
                optimizer="sgd",
                random_seed=4,
            ),
        )
        num_iterations = 5

        param_values = []
        for _ in range(num_iterations):
            pred = st.predict()

            reward = -1  # Arbitrary reward
            param_values.append(pred)

            st.set_reward(reward)

        for idx, values in enumerate(param_values):
            self.assertDictEqual(values, _EXPECTED_PARAM_VALUES_2[idx])

    def test_rmsprop(self):
        """Testing the RMSProp optimizer."""

        def get_reward(pred: np.ndarray):
            """Root Mean Squared Error."""
            target = np.asarray([0.1, 0.3, 0.6])
            return -np.sqrt(np.mean(np.square(pred - target)))

        _EXPECTED_PARAM_VALUES = [
            {"p1": 0.8781564936842099, "p2": 0.3472044439232318, "p3": 0.017909807169751893},
            {"p1": 0.8690912022386417, "p2": 0.25756734896540956, "p3": 0.19646795371739412},
            {"p1": 0.8125991917821089, "p2": 0.17888040096004929, "p3": 0.11814716484067767},
            {"p1": 0.7346488140848955, "p2": 0.22589140384952103, "p3": 0.2962305841255394},
            {"p1": 0.6051166016390045, "p2": 0.3336337318464624, "p3": 0.3292784421461223},
            {"p1": 0.7425687354276459, "p2": 0.20660798644577394, "p3": 0.25877049279040115},
            {"p1": 0.495967060797921, "p2": 0.4081768018428006, "p3": 0.3248072786429289},
            {"p1": 0.6716105635230628, "p2": 0.3134774344738803, "p3": 0.33827538060160844},
            {"p1": 0.5419375414893253, "p2": 0.44248568790724285, "p3": 0.23383662401343341},
            {"p1": 0.47533481168297703, "p2": 0.39042639953324787, "p3": 0.41509309114312176},
        ]

        parameters = (
            ContinuousParameter(
                name="p1",
                initial_value=0.87362385,
                constraint=Constraint(
                    lb=0.0,
                    ub=1.0,
                    dtype="float",
                ),
            ),
            ContinuousParameter(
                name="p2",
                initial_value=0.30238590,
                constraint=Constraint(
                    lb=0.0,
                    ub=1.0,
                    dtype="float",
                ),
            ),
            ContinuousParameter(
                name="p3",
                initial_value=0.10718888,
                constraint=Constraint(
                    lb=0.0,
                    ub=1.0,
                    dtype="float",
                ),
            ),
        )

        st = SelfTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback="twopoint",
                eta=0.01,
                delta=0.1,
                optimizer="rmsprop",
                random_seed=4,
            ),
        )

        num_iterations = 10

        param_values = []
        for _ in range(num_iterations):
            pred = st.predict()

            reward = get_reward(np.asarray([pred["p1"], pred["p2"], pred["p3"]]))
            param_values.append(pred)

            st.set_reward(reward)

        for idx, values in enumerate(param_values):
            for param in parameters:
                self.assertAlmostEqual(values[param.name], _EXPECTED_PARAM_VALUES[idx][param.name])
