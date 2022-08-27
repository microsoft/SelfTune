// ---------------------------------------------------------------------------
// <copyright file="SelfTune.cs" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// ---------------------------------------------------------------------------

namespace Microsoft.Research.SelfTune
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Type of feedback update.
    /// </summary>
    public enum Feedback : int
    {
        OnePoint = 1,
        TwoPoint = 2
    }

    /// <summary>
    /// Data-driven tuning of parameters towards values that produce larger rewards.
    /// </summary>
    public class SelfTune
    {
        private readonly TaskData taskData;
        private SessionData sessionData;

        private int round;

        /// <summary>
        /// The random number generator used by the algorithm.
        /// </summary>
        public Random random;

        /// <summary>
        /// A unique identifier for the point being returned in <see cref="Predict"/>.
        /// Changes after <see cref="SetReward(double)"/> (because a new point is returned
        /// then).
        /// </summary>
        public uint PredictId
        {
            get
            {
                return this.sessionData.FirstReward == null ? this.sessionData.FirstExploreId : this.sessionData.SecondExploreId;
            }
        }

        /// <summary>
        /// The point around which exploration is done.
        ///
        /// Note: This property provides some behind-the-scenes information for additional
        /// insight. Its existence can be safely ignored by the consumer of <see cref="SelfTune"/>.
        /// </summary>
        public IList<double> Center
        {
            get
            {
                int numParameters = this.taskData.InitialValues.Count;

                List<double> center = new List<double>(numParameters);

                for (int i = 0; i < numParameters; i++)
                {
                    center.Add(this.sessionData.Center[i]);
                }

                return center;
            }
        }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="initialValues">Initial values for the parameters.</param>
        /// <param name="constraints">Constraints on the parameters.</param>
        /// <param name="feedback">Type of feedback update to use.</param>
        /// <param name="eta">Hyperparameter for the algorithm - learning rate.</param>
        /// <param name="delta">Hyperparameter for the algorithm - exploration radius.</param>
        /// <param name="randomState">The random seed used to initialize the pseudo-random number generator.</param>
        /// <param name="etaDecayRate">The decay rate of eta. Defaults to 0 where eta does not decay after each round.</param>
        /// <exception cref="ArgumentException">When task related data isn't valid.</exception>
        public SelfTune(IList<double> initialValues, IList<Constraints> constraints, Feedback feedback, double eta, double delta, int? randomState = null, double etaDecayRate = 0.0)
        {
            TaskData taskData = new TaskData()
            {
                InitialValues = initialValues,
                Constraints = constraints,
                Feedback = feedback,
                Eta = eta,
                InitialEta = eta,
                Delta = delta,
                EtaDecayRate = etaDecayRate
            };
            SelfTune.ValidateTaskData(taskData);

            if (randomState == null)
            {
                this.random = new Random();
            }
            else
            {
                this.random = new Random(randomState.Value);
            }

            for (int i = 0; i < initialValues.Count; i++)
            {
                taskData.InitialValues[i] = SelfTune.Project(taskData.InitialValues[i], taskData.Constraints[i], taskData.Delta);
            }

            this.sessionData = SelfTune.GetNewSession(taskData.InitialValues, this.SampleUnitSphere(taskData.InitialValues.Count), taskData, 0);
            this.taskData = taskData;
        }

        /// <summary>
        /// Get the parameter values to use.
        /// Until <see cref="SetReward(double)"/> is called, calls to this will return the same value.
        /// </summary>
        /// <returns>The parameter values to use.</returns>
        public IList<double> Predict()
        {
            return this.sessionData.FirstReward == null ? this.sessionData.FirstExplore : this.sessionData.SecondExplore;
        }

        /// <summary>
        /// Notify the algorithm of how well the parameter values returned in <see cref="Predict"/> performed.
        /// </summary>
        /// <param name="reward">Measure of how well the parameter values returned in <see cref="Predict"/> performed.</param>
        public void SetReward(double reward)
        {
            if (this.taskData.Feedback == Feedback.OnePoint)
            {
                IList<double> newCenter = SelfTune.ComputeNewCenter(this.sessionData.Center, this.sessionData.ExploreDirection, reward, this.taskData);
                this.sessionData = SelfTune.GetNewSession(newCenter, this.SampleUnitSphere(newCenter.Count), this.taskData, this.sessionData.FirstExploreId + 1);
                this.round += 1;
            }
            else
            {
                if (!this.sessionData.FirstReward.HasValue)
                {
                    // reward for w + delta*u is now available
                    this.sessionData.FirstReward = reward;
                }
                else
                {
                    // reward for w - delta*u is now available, use this with the reward for w + delta*u to compute new center
                    IList<double> newCenter = SelfTune.ComputeNewCenter(this.sessionData.Center, this.sessionData.ExploreDirection, this.sessionData.FirstReward.Value - reward, this.taskData);
                    this.round += 1;
                    this.sessionData = SelfTune.GetNewSession(newCenter, this.SampleUnitSphere(newCenter.Count), this.taskData, this.sessionData.SecondExploreId + 1);
                }
            }
        }

        /// <summary>
        /// Sample uniformly from the surface of a unit sphere in n dimensions.
        /// </summary>
        private IList<double> SampleUnitSphere(int n)
        {
            List<double> sample = new List<double>(n);

            double norm = 0;
            for (int i = 0; i < n; i++)
            {
                double currentSample = this.SampleStandardNormal();
                norm += currentSample * currentSample;
                sample.Add(currentSample);
            }

            // norm == 0 is very unlikely, nevertheless handling it.
            if (norm == 0)
            {
                if (n > 0)
                {
                    sample[0] = (2 * this.random.Next(2)) - 1;
                }
            }
            else
            {
                norm = Math.Sqrt(norm);
                for (int i = 0; i < n; i++)
                {
                    sample[i] /= norm;
                }
            }

            return sample;
        }

        /// <summary>
        /// Sample from a standard normal distribution, i.e. ~ N(0, 1)
        /// </summary>
        private double SampleStandardNormal()
        {
            double u1 = 1 - this.random.NextDouble();
            double u2 = 1 - this.random.NextDouble();
            return Math.Sqrt(-2 * Math.Log(u1)) * Math.Sin(2 * Math.PI * u2);
        }

        /// <summary>
        /// Projects the center based on constraints.
        /// </summary>
        /// <param name="val">Value to project.</param>
        /// <param name="constraints">Constraints on the value.</param>
        /// <param name="delta">The exploration radius.</param>
        /// <returns>The projected set of parameters.</returns>
        private static double Project(double val, Constraints constraints, double delta)
        {
            double min = constraints.Min, max = constraints.Max;
            if (constraints.IsInt)
            {
                min = Math.Ceiling(min);
                max = Math.Floor(max);
                delta = Math.Ceiling(delta);
            }

            min += delta;
            max -= delta;

            val = Math.Min(Math.Max(min, val), max);
            if (constraints.IsInt)
            {
                val = Math.Round(val);
            }

            return val;
        }

        /// <summary>
        /// Decay eta based on the decay rate and current round.
        /// </summary>
        private double EtaUpdate()
        {
            return this.taskData.InitialEta / (1 + this.taskData.EtaDecayRate * this.round);
        }

        /// <summary>
        /// Compute the updated set of parameters using the reward(s) obtained during
        /// exploration around the current set of parameters.
        /// </summary>
        /// <param name="center">Current parameter vector.</param>
        /// <param name="exploreDirection">Direction of exploration.</param>
        /// <param name="rewardGradient">Contribution of the reward(s) towards the gradient.</param>
        /// <param name="taskData">Task for which new center is to be computed.</param>
        /// <returns>New set of parameters.</returns>
        private static IList<double> ComputeNewCenter(
            IList<double> center,
            IList<double> exploreDirection,
            double rewardGradient,
            TaskData taskData)
        {
            List<double> newCenter = new List<double>(center.Count);

            for (int i = 0; i < center.Count; i++)
            {
                newCenter.Add(
                    SelfTune.Project(
                        center[i] + (taskData.Eta * center.Count * rewardGradient * exploreDirection[i] / (((int)taskData.Feedback) * taskData.Delta)),
                        taskData.Constraints[i],
                        taskData.Delta));
            }

            return newCenter;
        }

        /// <summary>
        /// Compute the point to explore.
        /// </summary>
        /// <param name="center">The point around which exploration is done.</param>
        /// <param name="exploreSign">Whether to explore in the direction of exploreDirection or opposite to it.</param>
        /// <param name="exploreDirection">The direction along which to explore.</param>
        /// <param name="taskData">Task data.</param>
        /// <returns>Point to explore.</returns>
        private static IList<double> ComputeExploreValue(
            IList<double> center,
            ExploreSign exploreSign,
            IList<double> exploreDirection,
            TaskData taskData)
        {
            int numParameters = center.Count;
            List<double> exploreValue = new List<double>(numParameters);

            for (int i = 0; i < center.Count; i++)
            {
                double change = ((int)exploreSign) * taskData.Delta * exploreDirection[i];
                if (taskData.Constraints[i].IsInt)
                {
                    double rounded = Math.Round(change);
                    if (rounded == 0)
                    {
                        change = change >= 0 ? 1 : -1;
                    }
                    else
                    {
                        change = rounded;
                    }
                }
                exploreValue.Add(center[i] + change);
            }
            return exploreValue;
        }

        private static SessionData GetNewSession(
            IList<double> center,
            IList<double> exploreDirection,
            TaskData taskData,
            uint firstId)
        {
            return new SessionData
            {
                FirstExplore = SelfTune.ComputeExploreValue(center, ExploreSign.Positive, exploreDirection, taskData),
                SecondExplore = taskData.Feedback == Feedback.OnePoint ? null : SelfTune.ComputeExploreValue(center, ExploreSign.Negative, exploreDirection, taskData),
                Center = center,
                ExploreDirection = exploreDirection,
                FirstExploreId = firstId,
                SecondExploreId = taskData.Feedback == Feedback.OnePoint ? 0 : firstId + 1
            };
        }

        private static void ValidateTaskData(TaskData taskData)
        {
            int n = taskData.InitialValues.Count;

            if (taskData.InitialValues.Count != n)
            {
                throw new ArgumentException("Initial values not provided for all parameters");
            }

            if (taskData.Constraints.Count != n)
            {
                throw new ArgumentException("Constraints not provided for all parameters");
            }

            for (int i = 0; i < n; i++)
            {
                if (taskData.Constraints[i].Min > taskData.Constraints[i].Max)
                {
                    throw new ArgumentException($"Invalid constraint: Min(={taskData.Constraints[i].Min}) > Max({taskData.Constraints[i].Max})");
                }
            }

            if (taskData.Delta <= 0)
            {
                throw new ArgumentException($"Delta(={taskData.Delta}) should be > 0");
            }
        }
    }
}
