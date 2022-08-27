// ---------------------------------------------------------------------------
// <copyright file="SessionData.cs" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// ---------------------------------------------------------------------------

namespace Microsoft.Research.SelfTune
{
    using System.Collections.Generic;

    /// <summary>
    /// Whether exploration is in the direction of <see cref="SessionData.ExploreDirection"/>, 
    /// or opposite to it.
    /// </summary>
    internal enum ExploreSign : int
    {
        Positive = 1,
        Negative = -1
    }

    /// <summary>
    /// Data describing a session.
    /// </summary>
    public class SessionData
    {
        /// <summary>
        /// The 1st point to explore as part of this session.
        /// </summary>
        public IList<double> FirstExplore { get; set; }

        /// <summary>
        /// The 2nd point to explore as part of this session.
        /// (valid only for <see cref="Algorithm.TwoPoint"/>).
        /// </summary>
        public IList<double> SecondExplore { get; set; }

        /// <summary>
        /// Reward for <see cref="FirstExplore"/>.
        /// </summary>
        public double? FirstReward { get; set; }

        /// <summary>
        /// The point around which exploration is done.
        /// </summary>
        public IList<double> Center { get; set; }

        /// <summary>
        /// Direction along which to explore.
        /// </summary>
        public IList<double> ExploreDirection { get; set; }

        /// <summary>
        /// Identifier for <see cref="FirstExplore"/>.
        /// </summary>
        public uint FirstExploreId { get; set; }

        /// <summary>
        /// Identifier for <see cref="SecondExplore"/>.
        /// (valid only for <see cref="Algorithm.TwoPoint"/>).
        /// </summary>
        public uint SecondExploreId { get; set; }
    }
}
