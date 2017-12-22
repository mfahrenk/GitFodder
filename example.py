"""example.py

Compute the maximum of a Bessel function and plot it.

"""
import argparse

import numpy as np # alias numpy as np
from scipy import special, optimize # defines special and optimize in this script to belong
        						    # to scipy. An alertnative is to import scipy and the 
        						    # calls will require the module name, e.g.
        						    # sol = scipy.optimize.minimize(f,10)
import matplotlib.pyplot as plt

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("--order", type=int, default=3, help="order of Bessel function")
    parser.add_argument("--output", default="plot.png", help="output image file")
    args = parser.parse_args()

    # Compute maximum
    f = lambda x: -special.jv(args.order, x)
    sol = optimize.minimize(f, 1.0)

    # Plot
    x = np.linspace(0, 10, 5000)
    plt.plot(x, special.jv(args.order, x), '-', sol.x, -sol.fun, 'o')

    # Produce output
    plt.savefig(args.output, dpi=96)

if __name__ == "__main__":
    main()
