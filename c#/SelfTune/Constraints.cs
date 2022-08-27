// ---------------------------------------------------------------------------
// <copyright file="Constraints.cs" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// ---------------------------------------------------------------------------

namespace Microsoft.Research.SelfTune
{
    /// <summary>
    /// Constraints on a parameter learned using SelfTune.
    /// </summary>
    public class Constraints
    {
        /// <summary>
        /// Minimum value.
        /// </summary>
        public double Min { get; }

        /// <summary>
        /// Maximum value.
        /// </summary>
        public double Max { get; }

        /// <summary>
        /// Whether it is an integer.
        /// </summary>
        public bool IsInt { get; }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="min">Minimum value.</param>
        /// <param name="max">Maximum value.</param>
        /// <param name="isInt">Whether it is an integer.</param>
        public Constraints(double min = double.MinValue, double max = double.MaxValue, bool isInt = false)
        {
            this.Min = min;
            this.Max = max;
            this.IsInt = isInt;
        }
    }
}
