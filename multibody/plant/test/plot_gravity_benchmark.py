#!/usr/bin/env python3
"""Plot CalcGravityGeneralizedForces timings vs. N.

Reads the CSV produced by
//multibody/plant:calc_gravity_generalized_forces_benchmark and renders one
figure with two log-y curves (double, autodiff).

Usage:
    python3 plot_gravity_benchmark.py path/to/gravity_benchmark_timings.csv
    python3 plot_gravity_benchmark.py path.csv --output figure.png
"""

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="CSV file from the benchmark binary.")
    parser.add_argument(
        "--output",
        default=None,
        help="If given, save the figure here instead of calling show().",
    )
    args = parser.parse_args()

    series = defaultdict(lambda: ([], []))  # scalar_type -> (Ns, microseconds)
    with open(args.csv_path) as f:
        for row in csv.DictReader(f):
            xs, ys = series[row["scalar_type"]]
            xs.append(int(row["N"]))
            ys.append(float(row["min_time_microseconds"]))

    for scalar_type, (xs, ys) in sorted(series.items()):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(xs, ys, marker="o")

        ax.set_xlabel("N (number of links)")
        ax.set_ylabel("min CalcGravityGeneralizedForces time (microseconds)")
        
        ax.legend()
        ax.set_title(f"CalcGravityGeneralizedForces {scalar_type} scalability")

        if args.output:
            fig_name = f"{args.output}<{scalar_type}>.png"
            fig.savefig(fig_name, dpi=150, bbox_inches="tight")
        else:
            plt.show()


if __name__ == "__main__":
    main()
