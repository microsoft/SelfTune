namespace Driver
{
    using Microsoft.Research.SelfTune;
    using System;
    using System.Collections.Generic;

    class Program
    {
        static void Main(string[] args)
        {
            int numIterations = 100;

            // 1. Create the task
            double[] initialValues = new double[] { 0.4, 0.4 };
            Constraints[] constraints = new Constraints[] { new Constraints(min: 0, max: 1, isInt: false), new Constraints(min: 0, max: 1, isInt: false) };
            double eta = 0.01;
            double delta = 0.1;

            // 2. Create an instance of SelfTune to start learning
            SelfTune st = new SelfTune(initialValues, constraints, Feedback.TwoPoint, eta, delta);

            List<double> rewards = new List<double>(numIterations);

            // 3. Simulate the predict-setreward cycle
            for (int i = 0; i < numIterations; i++)
            {
                // 3. (a) Get prediction
                IList<double> prediction = st.Predict();

                // 3. (b) Set reward
                double reward = Reward(prediction[0], prediction[1]);
                st.SetReward(reward);
                rewards.Add(reward);

                if (i % 25 == 0)
                {
                    Console.WriteLine(
                        $"{i}: {reward:0.00000}, prediction=({prediction[0]:0.00000}, {prediction[1]:0.00000}, best=({st.Center[0]:0.00000}, {st.Center[1]:0.00000})"
                    );
                }
            }
        }

        static double Reward(double p1, double p2)
        {
            double targetp1 = 0.2, targetp2 = 0.8;
            return -((p1 - targetp1) * (p1 - targetp1) + (p2 - targetp2) * (p2 - targetp2));
        }
    }
}
