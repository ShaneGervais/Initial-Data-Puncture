from numpy import zeros, size, sqrt, linspace
from EllipticSolver import EllipticSolver
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, projections
import matplotlib.pyplot as plt
import numpy as np


"""
This class creates the initial data for our blackhole puncture.
Initial data is used at arguments and then we call the constructed 
solution
"""

class Puncture:

    """
    Arguments for the constructor:
        location of puncture -> bh_loc
        linear momentum -> lin_mom
        size of grid -> n_grid
        outer boundary -> x_out
    """
    def __init__(self, bh_loc, lin_mom, n_grid, x_out):

        #Initialize arguments
        self.bh_loc = bh_loc
        self.lin_mom = lin_mom
        self.n_grid = n_grid
        self.x_out = x_out
        self.delta = 2.0*x_out/n_grid

        #Echo parameters
        print("Constructing class Puncture for single black hole")
        print("     at location = (", bh_loc[0], ", ", bh_loc[1], ", ", bh_loc[2], ")")
        print("     with momentum p = (", lin_mom[0], ", ", lin_mom[1], ", ", lin_mom[2], ")")
        print("Using ", n_grid, "\b^3 gridpoints with outer boundar at ", x_out)

        half_delta = self.delta/2.0

        self.x = linspace(half_delta - x_out, x_out - half_delta, n_grid)
        self.y = linspace(half_delta - x_out, x_out - half_delta, n_grid)
        self.z = linspace(half_delta - x_out, x_out - half_delta, n_grid)

        #allocate the elliptic solver
        self.solver = EllipticSolver(self.x, self.y, self.z)


        #allocate functions u, alpha, beta and residual
        self.alpha = zeros((n_grid, n_grid, n_grid))
        self.beta = zeros((n_grid, n_grid, n_grid))
        self.u = zeros((n_grid, n_grid, n_grid))
        self.res = zeros((n_grid, n_grid, n_grid))

    #Construct our solutions
    def construct_solution(self, tol, it_max):

        self.setup_alpha_beta()
        residual_norm = self.residual()
        print("Initial Residual = ", residual_norm)
        print("Using up to ", it_max, "iteration steps to reach a tolerance of", tol)

        #iterate
        it_steps = 0
        
        while residual_norm > tol and it_steps < it_max:
            it_steps += 1
            self.update_u()
            residual_norm = self.residual()
            print("Residual after= ", it_steps, "iterations:", residual_norm)

        if (residual_norm < tol):
            print("Done!")

        else:
            print("Giving up")

    #Updates using poisson solver
    def update_u(self):
        
        #Initialize variables
        n_grid = self.n_grid
        fct = zeros((n_grid, n_grid, n_grid))
        rhs = zeros((n_grid, n_grid, n_grid))

        #h'(u^[n]) = 7*alpha*beta*(alpha + alpha*u^[n] +1)^-8
        for i in range(1, n_grid - 1):
            for j in range(1, n_grid - 1):
                for k in range(1, n_grid - 1):
                    temp = self.alpha[i, j, k]*(1.0 + self.u[i, j, k]) + 1.0
                    fct[i, j, k] = (-7.0*self.beta[i, j, k]*self.alpha[i, j, k]/ temp**8)
                    rhs[i, j, k] = -self.res[i, j, k]

        #update poisson solver
        self.solver.setup_matrix(fct)

        #setup rhs
        self.solver.setup_rhs(rhs)

        #solve for delta_u
        #i.e. D^2(delat_u) - h'(u^[n])(delta_u) = - R^[n]
        delta_u = self.solver.solve()

        #now update u
        self.u += delta_u

    #Calculate the residual
    #by R^[n] = (D^2)u^[n] - h(u^[n])
    def residual(self):

        #initialize
        residual_norm = 0.0
        n_grid = self.n_grid

        for i in range(1, n_grid - 1):
            for j in range(1, n_grid - 1):
                for k in range(1, n_grid - 1):

                    #lhs for laplace operator
                    ddx = (self.u[i + 1, j, k] - 2.0*self.u[i, j, k] + self.u[i - 1, j, k])
                    ddy = (self.u[i, j + 1, k] - 2.0*self.u[i, j, k] + self.u[i, j - 1, k])
                    ddz = (self.u[i, j, k + 1] - 2.0*self.u[i, j, k] + self.u[i, j, k - 1])

                    lhs = (ddx + ddy + ddz)/self.delta**2

                    #rhs
                    temp = self.alpha[i, j, k]*(1.0 + self.u[i, j, k]) + 1.0
                    rhs = -self.beta[i, j, k]/temp**7


                    self.res[i, j, k] = lhs - rhs
                    residual_norm += self.res[i, j, k]**2

        residual_norm = sqrt(residual_norm) * self.delta**3
        return residual_norm


    def setup_alpha_beta(self):
        n_grid = self.n_grid

        #Initialize momentum with given arguments
        pX = self.lin_mom[0]
        pY = self.lin_mom[1]
        pZ = self.lin_mom[2]

        for i in range(0, n_grid):
            for j in range(0, n_grid):
                for k in range(0, n_grid):
                    sX = self.x[i] - self.bh_loc[0]
                    sY = self.y[i] - self.bh_loc[1]
                    sZ = self.z[i] - self.bh_loc[2]
                    s2 = sX**2 + sY**2 + sZ**2
                    s_bh = sqrt(s2)

                    lX = sX/s_bh
                    lY = sY/s_bh
                    lZ = sZ/s_bh

                    lP = lX*pX + lY*pY + lZ*pZ

                    #Construct curvature
                    #A^[i,j]_L = (3/2s^2)((P^i)(l^j) + (P^j)(l^i) - (n^ij -(l^i)(l^j))l_k(P^k))
                    fac = 3.0/(2.0*s2)
                    
                    Axx = fac*(2.0*pX*lX - (1.0 - lX*lX)*lP)
                    Ayy = fac*(2.0*pY*lY - (1.0 - lY*lY)*lP)
                    Azz = fac*(2.0*pZ*lZ - (1.0 - lZ*lZ)*lP)

                    Axy = fac*(pX*lY + pY*lX + lX*lY*lP)
                    Axz = fac*(pX*lZ + pZ*lX + lX*lZ*lP)

                    Ayz = fac*(pY*lZ + pZ*lY + lY + lY * lZ *lP)

                    #Compute A
                    A2 = (Axx**2 + Ayy**2 + Azz**2 + 2.0*(Axy**2 + Axz**2 + Ayz**2))


                    #now for alpha and beta
                    #1/alpha = sum(M_n/2s_n)
                    #beta = (1/8)(alpha^7)A^L_ij*A^ij_L
                    self.alpha[i, j, k] = 2.0*s_bh
                    self.beta[i, j, k] = self.alpha[i, j, k]**7 * A2/8.0

    #end of setup for alpha and beta

    #output to a file
    def write_to_file(self):

        n_grid = self.n_grid
        x_out = self.x_out
        bh_loc = self.bh_loc
        lin_mom = self.lin_mom
        
        #create a file
        filename = "Puncture_" + str(n_grid) + "_" + str(x_out)
        filename = filename + ".data"
        out = open(filename, "w")

        if out:
            k = n_grid//2
            out.write("#Data for black hole at x (%f,%f,%f)\n" % (bh_loc[0], bh_loc[1], bh_loc[2]))

            out.write("#with linear momentum p = (%f,%f,%f)\n" % (lin_mom))
            out.write("#in plane for z = %e \n" % (self.z[k]))
            out.write("#x            y              u               \n")
            out.write("#============================================\n")

            for i in range(0, n_grid):
                for j in range(0, n_grid):
                    out.write("%e  %e  %e\n" % (self.x[i], self.y[j], self.u[i, j, k]))
                    out.write("\n")
                    
            out.close()
        
        else:
            print("Could not open file", filename, "in write_to_file()")
            print("permission error?")
    #end write file

#end class