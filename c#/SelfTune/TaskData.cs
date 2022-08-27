// ---------------------------------------------------------------------------
// <copyright file="TaskData.cs" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// ---------------------------------------------------------------------------

namespace Microsoft.Research.SelfTune
{
    using System.Collections.Generic;

    /// <summary>
    /// Data representing the learning task.
    /// </summary>
    public class TaskData
    {
        /// <summary>
        /// Initial values for the parameters.
        /// </summary>
        public IList<double> InitialValues { get; set; }

        /// <summary>
        /// Constraints on the parameters.
        /// </summary>
        public IList<Constraints> Constraints { get; set; }

        /// <summary>
        /// Type of feedback update.
        /// </summary>
        public Feedback Feedback { get; set; }

        /// <summary>
        /// Learning rate in the ongoing round.
        /// </summary>
        public double Eta { get; set; }

        /// <summary>
        /// The initial value of the learning rate.
        /// <summary>
        public double InitialEta { get; set; }

        /// <summary>
        /// Exploration radius.
        /// </summary>
        public double Delta { get; set; }

        /// <summary>
        /// The decay rate of eta.
        /// </summary>
        public double EtaDecayRate { get; set; }
    }
}
