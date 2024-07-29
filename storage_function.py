from SumOfSquares import SOSProblem, poly_opt_prob
import sympy as sp
from itertools import combinations_with_replacement, product
import numpy as np

alpha, beta, k = 1, 1, 1

# Define a polynomial with degree n in sympy:
degree = 2

num_nodes = 3
state_dim = 2

A = np.array([[0, 1, 1],
              [1, 0, 0],
              [0, 1, 0]])


def generate_general_polynomial(n, d):
    # Define the variables x1, x2, ..., xn
    x = sp.symbols(f'x1:{n + 1}')

    # List to hold the terms of the polynomial
    polynomial_terms = []

    # Set to collect all coefficient symbols
    coefficients = set()

    # Loop over all degrees from 0 to d
    for degree in range(d + 1):
        # Generate all possible monomials of the current degree
        for powers in combinations_with_replacement(range(n), degree):
            # Compute the monomial x1^a1 * x2^a2 * ... * xn^an
            monomial = sp.Mul(*[x[i] for i in powers])

            # Define a new symbol for the coefficient of this monomial
            coeff = sp.symbols(f'a{"_".join(map(str, sorted(powers)))}')

            # Add the coefficient to the set
            coefficients.add(coeff)

            # Add the term (coefficient * monomial) to the polynomial terms
            polynomial_terms.append(coeff * monomial)

    # Combine all terms to form the polynomial
    polynomial = sp.Add(*polynomial_terms)

    return list(coefficients), x, polynomial


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
