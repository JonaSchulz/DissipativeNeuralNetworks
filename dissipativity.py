from SumOfSquares import SOSProblem
import sympy as sp
from sympy import lambdify
from itertools import combinations_with_replacement
import numpy as np


class NodeDynamics:
    def __init__(self, alpha=1, beta=1, k=1):
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.x1, self.x2, self.u = sp.symbols('x1 x2 u')
        self.input = sp.Matrix([self.u])
        self.output = sp.Matrix([self.x2])
        self.state = sp.Matrix([self.x1, self.x2])

    def __call__(self):
        return sp.Matrix([self.x2, -self.alpha * self.x1 ** 3 - self.k * self.x2 + self.beta * self.u])


class L2Gain:
    def __init__(self):
        self.gamma = sp.symbols('g')
        self.gamma_value = None

    def __call__(self, input, output):
        return self.gamma * input.dot(input) - output.dot(output)

    def evaluate(self, u, y, gamma_value):
        return gamma_value * u ** 2 - y ** 2


class Passivity:
    def __call__(self, input, output):
        assert input.shape == output.shape, 'Input and output dimensions must match'
        return input.dot(output)

    def evaluate(self, u, y):
        return u * y


class NoSupply:
    def __call__(self, input, output):
        return 0


class Dissipativity:
    def __init__(self, dynamics, supply_rate, degree):
        self.dynamics = dynamics
        self.input, self.output, self.state = dynamics.input, dynamics.output, dynamics.state
        self.state_dim = len(self.state)
        self.supply_rate = supply_rate
        self.degree = degree
        self.coefficient_values = None
        if isinstance(self.supply_rate, L2Gain):
            self.gamma_value = None

        self.coefficients, self.polynomial = self.generate_general_polynomial(self.state, degree)
        self.gradient = self.compute_gradient()

        # Non-negativity inequality:
        self.non_negativity = self.polynomial

        # Dissipativity inequality:
        self.dissipativity = -sp.Matrix(self.gradient).dot(self.dynamics()) + self.supply_rate(self.input, self.output)

    def generate_general_polynomial(self, x, degree):
        # List to hold the terms of the polynomial
        polynomial_terms = []

        # Set to collect all coefficient symbols
        coefficients = set()

        # Loop over all degrees from 0 to d
        for d in range(degree + 1):
            # Generate all possible monomials of the current degree
            for powers in combinations_with_replacement(range(self.state_dim), d):
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

        return list(coefficients), polynomial

    def compute_gradient(self):
        # Compute the gradient of the polynomial
        gradient = sp.Matrix([sp.diff(self.polynomial, var) for var in self.state])
        return gradient

    def find_storage_function(self):
        sos_prob = SOSProblem()
        sos_prob.add_sos_constraint(self.non_negativity, list(self.state) + list(self.input))
        sos_prob.add_sos_constraint(self.dissipativity, list(self.state) + list(self.input))
        if isinstance(self.supply_rate, L2Gain):
            gamma_v = sos_prob.sym_to_var(self.supply_rate.gamma)
            sos_prob.add_sos_constraint(self.supply_rate.gamma, list(self.state) + list(self.input))
            sos_prob.set_objective("min", gamma_v)
        sos_prob.solve(solver="mosek")
        coefficients_v = [sos_prob.sym_to_var(i) for i in self.coefficients]
        self.coefficient_values = [i.value for i in coefficients_v]

        if isinstance(self.supply_rate, L2Gain):
            self.supply_rate.gamma_value = gamma_v.value
            return self.coefficient_values, gamma_v.value

        return self.coefficient_values

    def evaluate_storage(self, x, coefficient_values):
        # Evaluate the polynomial for x given by an array of shape [time_steps, num_nodes, state_dim]
        time_steps, num_nodes, state_dim = x.shape
        x = x.reshape(time_steps * num_nodes, state_dim)
        polynomial = lambdify(list(self.state), self.polynomial.subs({coeff: value for coeff, value in zip(self.coefficients, coefficient_values)}))
        return polynomial(*x.T).reshape(time_steps, num_nodes)

    def evaluate_dissipativity(self, x, u):
        # Evaluate the dissipativity inequality for x given by an array of shape [time_steps, num_nodes, state_dim]
        if len(x.shape) > 3:
            x = x.reshape(-1, x.shape[2], x.shape[3])
        time_steps, num_nodes, state_dim = x.shape
        x = x.reshape(time_steps * num_nodes, state_dim)
        u = u.reshape(time_steps * num_nodes, 1)
        coeff_dict = {coeff: value for coeff, value in zip(self.coefficients, self.coefficient_values)}
        if isinstance(self.supply_rate, L2Gain):
            coeff_dict[self.supply_rate.gamma] = self.supply_rate.gamma_value
        gradient = lambdify(list(self.state) + list(self.input), self.dissipativity.subs(coeff_dict))
        return gradient(*x.T, *u.T).reshape(time_steps, num_nodes)


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
