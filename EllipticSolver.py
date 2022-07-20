from numpy import zeros, size, sqrt
import scipy.linalg as la

"""
Class to solve a poisson type elliptic equation
with form D^2 sol + fct sol = rhs 
where 
    D^2 is the flat laplace operator 
    fct and rhs are user-supplied functions of x,y,z
    sol is the solution
"""
class EllipticSolver:
    
    def __init__(self, x, y, z):

        print("Setting up the Poisson solver")

        self.n_grid = size(x)
        self.delta = x[1] - x[0]

        #Number of Grids
        n_grid = self.n_grid

        #Create 3D grid
        nnn = n_grid ** 3
        
        #1D RHS of the equ
        self.rhs_1d = zeros(nnn)

        #Create 3x3 matrix A
        self.A = zeros((nnn, nnn))
        
        self.sol = zeros((n_grid, n_grid, n_grid))
        self.rad = zeros((n_grid, n_grid, n_grid))

        #Compute the radius
        for i in range(0, n_grid):
            for j in range(0, n_grid):
                for k in range (0, n_grid):
                    rad2 = x[i]**2 + y[j]**2 + z[k]**2
                    self.rad[i, j, k] = sqrt(rad2) 
    #enf of init

    #Initializing matrix A
    def setup_matrix(self, fct):

        #Number of Grids
        n_grid = self.n_grid

        #Using Robin boundary conditions for BC
        
        #In X

        #Lower boundary
        i = 0 
        for j in range(0, n_grid):
            for k in range(0, n_grid):
                index = self.super_index(i, j, k)
                self.A[index, index] = self.rad[i, j, k]
                self.A[index, index + 1] = -self.rad[i + 1, j, k]

        #Upper Boundary
        i = n_grid - 1
        for j in range(0, n_grid):
            for k in range(0, n_grid):
                index = self.super_index(i, j, k)
                self.A[index, index] = self.rad[i, j, k]
                self.A[index, index - 1] = -self.rad[i - 1, j, k]

        #In Y

        #Lower boundary
        j = 0 
        for i in range(1, n_grid - 1):
            for k in range(0, n_grid):
                index = self.super_index(i, j, k)
                self.A[index, index] = self.rad[i, j, k]
                self.A[index, index + n_grid] = -self.rad[i, j + 1, k]

        #Upper Boundary
        j = n_grid - 1
        for i in range(1, n_grid - 1):
            for k in range(0, n_grid):
                index = self.super_index(i, j, k)
                self.A[index, index] = self.rad[i, j, k]
                self.A[index, index - n_grid] = -self.rad[i, j - 1, k]

        #In Z

        #Lower boundary
        k = 0 
        for i in range(1, n_grid - 1):
            for j in range(1, n_grid - 1):
                index = self.super_index(i, j, k)
                self.A[index, index] = self.rad[i, j, k]
                self.A[index, index + n_grid * n_grid] = -self.rad[i, j, k + 1]

        #Upper Boundary
        k = n_grid - 1
        for i in range(1, n_grid - 1):
            for j in range(1, n_grid - 1):
                index = self.super_index(i, j, k)
                self.A[index, index] = self.rad[i, j, k]
                self.A[index, index - n_grid * n_grid] = -self.rad[i, j, k - 1]

        #Filling matrix
        for i in range(1, n_grid - 1):
            for j in range(1, n_grid - 1):
                for k in range(1, n_grid - 1):
                    
                    index = self.super_index(i, j, k)

                    #diagonal
                    self.A[index, index] = -6. + self.delta**2 *fct[i, j, k]

                    #off diagonal
                    self.A[index, index - 1] = 1.0
                    self.A[index, index + 1] = 1.0
                    self.A[index, index - n_grid] = 1.0
                    self.A[index, index + n_grid] = 1.0
                    self.A[index, index - n_grid * n_grid] = 1.0
                    self.A[index, index + n_grid * n_grid] = 1.0

    #matrix A setup end

    #initialize RHS of equation
    def setup_rhs(self, rhs):
        
        n_grid = self.n_grid

        for i in range(1, n_grid - 1):
            for j in range(1, n_grid - 1):
                for k in range(1, n_grid - 1):
                    index = self.super_index(i, j, k)
                    self.rhs_1d[index] = self.delta**2 *rhs[i, j, k]

    #Uses scipy.linalg matrix solver and return our sol in 3D
    def solve(self):
        
        #solve in 1D
        sol_1d = la.solve(self.A, self.rhs_1d)

        #then

        #translate to 3D
        for i in range(0, self.n_grid):
            for j in range(0, self.n_grid):
                for k in range(0, self.n_grid):
                    index = self.super_index(i, j, k)
                    self.sol[i, j, k] = sol_1d[index]

        return self.sol

    #Using super indices i + Nj + N^2k
    def super_index(self, i, j, k):
        return i + self.n_grid * (j + self.n_grid*k)