from SumOfSquares import SOSProblem, poly_opt_prob
import sympy as sp
from itertools import combinations_with_replacement, product
import numpy as np
import warnings
import cvxpy as cp # SDP library

alpha, beta, k = 1, 1, 1

# Define a polynomial with degree n in sympy:
degree = 2

num_nodes = 3
state_dim = 2

A = np.array([[0, 1, 1],
              [1, 0, 0],
              [0, 1, 0]])


def generate_general_polynomial(n, s, d):

    # 2D numpy array of variables - nxs
    x_matrix = np.empty((n, s), dtype=object)
    for i in range(n):
        for j in range(s):
            x_matrix[i, j] = sp.symbols(f'x{i+1}{j+1}')

    # List to hold the terms of the polynomial for all nodes
    polynomial_terms = []

    # Set to collect all coefficient symbols
    coefficients = []

    # Generate monomial terms (common across all nodes)
    for degree in range(d + 1):
        # Generate all possible monomials of the current degree for two variables
        for powers in combinations_with_replacement(range(2), degree):
            # Define a new symbol for the coefficient of this monomial if not already defined
            coeff = sp.symbols(f'a{"_".join(map(str, sorted(powers)))}')
            coefficients.append(coeff)

            # For each node, compute the monomial using the same coefficient
            for node_index in range(n):
                # Get the variables for the current node
                node_vars = x[node_index]

                # Compute the monomial for the node
                monomial = sp.Mul(*[node_vars[i] for i in powers])

                # Add the term (coefficient * monomial) to the node terms
                polynomial_terms.append(coeff * monomial)

    # Combine all terms to form the polynomial
    polynomial = sp.Add(*polynomial_terms)

    return coefficients, x, polynomial


def compute_gradient(polynomial, variables):
    # Compute the gradient of the polynomial
    gradient = sp.Matrix([sp.diff(polynomial, var) for var in variables])
    return gradient


def generate_system_expressions(n, a, b, k, A, x):
    # List to hold the expressions for the derivatives
    derivatives = []

    # Construct the expressions for each system
    for i in range(n):
        # Variables for the current system
        x_i1 = x[2 * i]  # x_{i1}
        x_i2 = x[2 * i + 1]  # x_{i2}

        # Expression for \dot{x_{i1}} = x_{i2}
        dot_x_i1 = x_i2

        # Expression for \dot{x_{i2}}
        interaction_sum = b * sum(A[i, j] * (x[2 * j + 1] - x_i2) for j in range(n))
        dot_x_i2 = -a * x_i1 ** 3 - k * x_i2 + interaction_sum

        # Add the expressions to the list
        derivatives.extend([dot_x_i1, dot_x_i2])

    return derivatives


def compute_dV_dx_T_dot_x(V, derivatives, x):
    # Compute the gradient dV/dx
    gradient_V = sp.Matrix([sp.diff(V, var) for var in x])

    # Compute the dot product (dV/dx)^T * dot{x}
    dot_product = gradient_V.dot(sp.Matrix(derivatives))

    return dot_product


coefficients, x, p1 = generate_general_polynomial(state_dim * num_nodes, degree)
gradient_V = compute_gradient(p1, x)
x_dot = generate_system_expressions(num_nodes, alpha, beta, k, A, x)
p2 = -sp.Matrix(gradient_V).dot(sp.Matrix(x_dot))
print(x_dot)
print(gradient_V)
print(p1)
print(p2)


sos_prob = SOSProblem()
sos_prob.add_sos_constraint(p1, x)
sos_prob.add_sos_constraint(p2, x)

# gv = sos_prob.sym_to_var(g)
# av = sos_prob.sym_to_var(a)
# bv = sos_prob.sym_to_var(b)

coefficients_v = [sos_prob.sym_to_var(i) for i in coefficients]
#sos_prob.set_objective('min')
sos_prob.solve(solver="mosek")

#print([i.value for i in coefficients_v])
print([i.value for i in coefficients_v])


class Stability:
    valid_dissipativities = ['P', 'OSP', 'ISP', 'L2'] # Passive, Output-strict passive, Input-strict passive, L2 norm

    def __init__(self, M, s, dissipativity, gamma):
        self.M = M # Connection matrix
        self.n = M.shape[0] # Nodes
        self.s = s # States

        if dissipativity not in self.valid_dissipativities:
            warnings.warn(f"'{dissipativity}' not a valid supply rate", UserWarning)
        self.dissipativity = dissipativity
        self.gamma = gamma # Only needed for OSP, ISP, L2

    def construct_QRS(self):
        if self.dissipativity == 'P': # Passive 
            Q = np.zeros((self.s, self.s))
            S = (1/2)*np.eye(self.n)
            R = Q

        elif self.dissipativity == 'OSP': # Output-strict passive
            Q = np.zeros((self.s, self.s))
            S = (1/2)*np.eye(self.n)
            R = -self.gamma**2*np.eye(self.n)

        elif self.dissipativity == 'ISP': # Input-strict passive
            Q = -self.gamma**2*np.eye(self.n)
            S = (1/2)*np.eye(self.n)
            R = np.zeros((self.s, self.s))

        elif self.dissipativity == 'L2': # L2 norm
            Q = self.gamma**2*np.eye(self.n)
            S = np.zeros((self.s, self.s))
            R = -np.eye(self.s)

        return Q, S, R
    
    def sdp_solve(self):
        # CVXPY variables
        p = cp.Variable(self.n) # p1, p2, ..., pn

        # Q, S, R matrices
        Q, S, R = self.construct_QRS()

        # Define bigger [Q, S; S' R] matrix
        block_Q = np.block([[p[i]*Q if i == j else np.zeros((self.s, self.s)) for j in range(self.n)] for i in range(self.n)]) # Big Q
        block_S = np.block([[p[i]*S if i == j else np.zeros((self.s, self.s)) for j in range(self.n)] for i in range(self.n)]) # Big S
        block_ST = np.block([[p[i]*S.T if i == j else np.zeros((self.s, self.s)) for j in range(self.n)] for i in range(self.n)]) # Big S'
        block_R = np.block([[p[i]*R if i == j else np.zeros((self.s, self.s)) for j in range(self.n)] for i in range(self.n)]) # Big R

        QS = np.concatenate((block_Q, block_S), axis = 1)
        STR = np.concatenate((block_ST, block_R), axis = 1)

        QSSR = np.vstack((QS, STR))

        # Define MI matrix
        I = np.eye(self.M.shape[0])
        MI = np.vstack((self.M, I))

        # DSP matrix
        stability_matrix = np.dot(MI.T, np.dot(QSSR, MI))

        # DSP problem
        constraints = [-stability_matrix >> 0]
        constraints.append(p > 0)
        objective = cp.Minimize(cp.sum(p))

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()

        return p.value











        
