from SumOfSquares import SOSProblem
import sympy as sp
from sympy import lambdify
from itertools import combinations_with_replacement
import numpy as np
import torch


class NonlinearOscillatorNodeDynamics:
    def __init__(self, a1=0.2, a2=11.0, a3=11.0, a4=1.0):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.x1, self.x2, self.u = sp.symbols('x1 x2 u')
        self.input = sp.Matrix([self.u])
        self.output = sp.Matrix([self.x2])
        self.state = sp.Matrix([self.x1, self.x2])

    def __call__(self):
        return sp.Matrix([self.x2, -self.x1 - self.a1 * self.x2 * (self.a2 * self.x1 ** 4 - self.a3 * self.x1 + self.a4) + self.u])

    def compute_u(self, x, adjacency_matrix):
        """
        Compute the control input u for the dissipativity evaluation (u_i = sum_j A_ij * (x_j[2] - x_i[2]))

        :param x: shape (time_steps, num_nodes, 2)
        :param adjacency_matrix: shape (num_nodes, num_nodes)
        :return: shape (time_steps, num_nodes, 1)
        """

        u_values = []
        for t in range(x.shape[0]):
            u_t = torch.sum(adjacency_matrix * (x[t, :, 1].reshape(-1, 1) - x[t, :, 1].reshape(1, -1)), dim=1)
            u_values.append(u_t)
        return torch.stack(u_values).unsqueeze(-1)


class NonlinearOscillator2NodeDynamics:
    def __init__(self, alpha=1, beta=1, k=1, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.x1, self.x2, self.u = sp.symbols('x1 x2 u')
        self.input = sp.Matrix([self.u])
        self.output = sp.Matrix([self.x2])
        self.state = sp.Matrix([self.x1, self.x2])

    def __call__(self):
        return sp.Matrix([self.x2, -self.alpha * self.x1 ** 3 - self.k * self.x2 + self.beta * self.u])

    def compute_u(self, x, adjacency_matrix):
        """
        Compute the control input u for the dissipativity evaluation (u_i = sum_j A_ij * (x_j[2] - x_i[2]))

        :param x: shape (time_steps, num_nodes, 2)
        :param adjacency_matrix: shape (num_nodes, num_nodes)
        :return: shape (time_steps, num_nodes, 1)
        """

        u_values = []
        for t in range(x.shape[0]):
            u_t = -torch.sum(adjacency_matrix * (x[t, :, 1].reshape(-1, 1) - x[t, :, 1].reshape(1, -1)), dim=1)
            u_values.append(u_t)
        return torch.stack(u_values).unsqueeze(-1)


class LotkaVolterraNodeDynamics:
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.x1, self.x2, self.u1, self.u2 = sp.symbols('x1 x2 u1 u2')
        self.input = sp.Matrix([self.u1, self.u2])
        self.output = sp.Matrix([self.x1, self.x2])
        self.state = sp.Matrix([self.x1, self.x2])

    def __call__(self):
        return sp.Matrix([self.alpha * self.x1 - self.beta * self.x1 * self.x2 + self.u1,
                          self.delta * self.x1 * self.x2 - self.gamma * self.x2 + self.u2])



class L2Gain:
    def __init__(self):
        self.gamma = sp.symbols('g')
        self.gamma_value = None

    def __call__(self, input, output):
        return self.gamma * input.dot(input) - output.dot(output)

    def evaluate(self, u, y, gamma_value):
        return gamma_value * u ** 2 - y ** 2

    def eval_string(self, string):
        return string.replace(str(self.gamma), str(self.gamma_value))


class Passivity:
    def __call__(self, input, output):
        assert input.shape == output.shape, 'Input and output dimensions must match'
        return input.dot(output)

    def evaluate(self, u, y):
        return u * y

    def eval_string(self, string):
        return string


class NoSupply:
    def __call__(self, input, output):
        return 0


class Dissipativity:
    def __init__(self, dynamics, supply_rate, degree):
        self.dynamics = dynamics
        self.input, self.output, self.state = dynamics.input, dynamics.output, dynamics.state
        self.state_dim = len(self.state)
        self.input_dim = len(self.input)
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

        self.x_dot_pred = [sp.symbols(f'x{i+1}_dot') for i in range(self.state_dim)]
        self.V_dot_pred = self.gradient.dot(sp.Matrix(self.x_dot_pred))
        self.dissipativity_pred = -self.V_dot_pred + self.supply_rate(self.input, self.output)

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

        # Sort coefficient by name length to avoid problems during evaluation of the polynomial
        coefficients = sorted(list(coefficients), key=lambda coeff: -len(str(coeff)))

        return coefficients, polynomial

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
        sos_prob.solve()
        coefficients_v = [sos_prob.sym_to_var(i) for i in self.coefficients]
        self.coefficient_values = [i.value for i in coefficients_v]

        if isinstance(self.supply_rate, L2Gain):
            self.supply_rate.gamma_value = gamma_v.value
            return self.coefficient_values, gamma_v.value

        return self.coefficient_values

    def evaluate_storage(self, x):
        # Evaluate the polynomial for x given by an array of shape [time_steps, num_nodes, state_dim]
        time_steps, num_nodes, state_dim = x.shape
        x = x.reshape(time_steps * num_nodes, state_dim)
        polynomial = lambdify(list(self.state), self.polynomial.subs({coeff: value for coeff, value in zip(self.coefficients, self.coefficient_values)}))
        return polynomial(*x.T).reshape(time_steps, num_nodes)

    def evaluate_dissipativity(self, x, u, x_dot):
        # Evaluate the dissipativity inequality for x given by an array of shape [time_steps, num_nodes, state_dim]
        assert x.shape[-1] == self.state_dim, 'State dimension does not match'
        assert u.shape[-1] == self.input_dim, 'Input dimension does not match'

        dissipativity_str = str(self.dissipativity_pred)
        for coeff, value in zip(self.coefficients, self.coefficient_values):
            dissipativity_str = dissipativity_str.replace(str(coeff), str(value))

        for i, x_dot_symbol in enumerate(self.x_dot_pred):
            dissipativity_str = dissipativity_str.replace(str(x_dot_symbol), f'x_dot[:, :, {i}]')

        for i, state in enumerate(self.state):
            dissipativity_str = dissipativity_str.replace(str(state), f'x[:, :, {i}]')

        for i, input in enumerate(self.input):
            dissipativity_str = dissipativity_str.replace(str(input), f'u[:, :, {i}]')

        dissipativity_str = self.supply_rate.eval_string(dissipativity_str)

        return eval(dissipativity_str)


class PendulumDynamics:
    def __init__(self, d, m, k):
        self.d = d
        self.m = m
        self.k = k

    def compute_u(self, x, adjacency_matrix):
        """
        Compute the control input u for the dissipativity evaluation (u_i = sum_j A_ij * (x_j[2] - x_i[2]))

        :param x: shape (time_steps, num_nodes, 2)
        :param adjacency_matrix: shape (num_nodes, num_nodes)
        :return: shape (time_steps, num_nodes, 1)
        """

        u_values = []
        for t in range(x.shape[0]):
            u_t = 1 / self.m * torch.sum(adjacency_matrix * torch.sin(x[t, :, 0].reshape(-1, 1) - x[t, :, 0].reshape(1, -1)), dim=1)
            u_values.append(u_t)
        return torch.stack(u_values).unsqueeze(-1)


class DissipativityPendulum:
    def __init__(self, d=1.0, m=1.0, k=1.0, **kwargs):
        self.d = d
        self.m = m
        self.k = k
        self.dynamics = PendulumDynamics(d, m, k)

    def evaluate_storage(self, x):
        return self.m * x[:, :, 1] ** 2 / 2 + self.k * (1 - torch.cos(x[:, :, 0]))

    def evaluate_dissipativity(self, x, u, x_dot):
        return x[:, :, 1] * u[:, :, 0] - self.k * torch.sin(x[:, :, 0]) * x_dot[:, :, 0] - self.m * x[:, :, 1] * x_dot[:, :, 1]
