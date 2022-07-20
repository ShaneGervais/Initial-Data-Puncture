"""
BlackholePunctures by Shane Gervais
last updated: 2022/07/18

This code is the main driver for EllipticSolver.py and Puncture.py
to construct the initial puncture data of a singular
blackhole. 

Use flag -h for list of options

These files are referenced by Numerical Relativity Starting 
from Scratch by Thomas W. Baumgarte and Stuart L. Shapiro
"""

import sys
from Puncture import Puncture

#main driver
def main():

    print("======================================================================")
    print("Blackhole initial data puncture, use flag -h for options")
    print("======================================================================")

    #set default values
    locX = 0.0
    locY = 0.0
    locZ = 0.0

    #momentum of blackhole
    pX = 1.0
    pY = 1.0
    pZ = 1.0

    #number of grids
    n_grid = 16

    #location of outer boundary
    x_out = 4.0

    #tolerance and maximum number of iterations
    tol = 1.0e-12
    it_max = 50

    #flag options
    for i in range(len(sys.argv)):

        if sys.argv[i] == "-h":
            usage()
            return

        if sys.argv[i] == "-n_grid":
            n_grid = int(sys.argv[i+1])

        if sys.argv[i] == "-x_out":
            x_out = float(sys.argv[i+1])

        if sys.argv[i] == "-locX":
            locX = float(sys.argv[i+1])

        if sys.argv[i] == "-locY":
            locY = float(sys.argv[i+1])

        if sys.argv[i] == "-locZ":
            locZ = float(sys.argv[i+1])

        if sys.argv[i] == "-pX":
            pX = float(sys.argv[i+1])

        if sys.argv[i] == "-pY":
            pY = float(sys.argv[i+1])

        if sys.argv[i] == "-pZ":
            pZ = float(sys.argv[i+1])

        if sys.argv[i] == "-tol":
            tol = float(sys.argv[i+1])

        if sys.argv[i] == "-it_max":
            it_max = int(sys.argv[i+1])

    #location of puncture
    bh_loc = (locX, locY, locZ)

    #linear momentum
    lin_mom = (pX, pY, pZ)

    #puncture solver
    blackHole = Puncture(bh_loc, lin_mom, n_grid, x_out)

    #Construct solution
    blackHole.construct_solution(tol, it_max)

    #write to file
    blackHole.write_to_file()

    #plot results
    blackHole.plot()

def usage():
    print("Construct the initial puncture data for a single black hole")
    print("Options are:")
    print("to change number of grids with -n_grid")
    print("to change location of outer boundary with -x_out")
    print("to change tolerance with -tol")
    print("to change maximum iterations with -it_max")
    print("to change location of x, y or z with -loc along with position X,Y or Z")
    print("to change momentum of x, y or z with -p along with position X,Y or Z")
    print("Otherwise the program will follow default values")

if __name__ == '__main__':
    main()