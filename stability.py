import numpy as np
import cvxpy as cp # SDP library
import warnings

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
            Q = cp.Parameter((self.s, self.s), value=np.zeros((self.s, self.s)))
            S = cp.Parameter((self.s, self.s), value=(1/2)*np.eye(self.s)) 
            R = Q

        elif self.dissipativity == 'OSP': # Output-strict passive
            Q = cp.Parameter((self.s, self.s), value=np.zeros((self.s, self.s)))
            S = cp.Parameter((self.s, self.s), value=(1/2)*np.eye(self.s)) 
            R = cp.Parameter((self.s, self.s), value=-self.gamma**2*np.eye(self.s)) 

        elif self.dissipativity == 'ISP': # Input-strict passive
            Q = cp.Parameter((self.s, self.s), value=-self.gamma**2*np.eye(self.s)) 
            S = cp.Parameter((self.s, self.s), value=(1/2)*np.eye(self.s)) 
            R = cp.Parameter((self.s, self.s), value=np.zeros((self.s, self.s)))

        elif self.dissipativity == 'L2': # L2 norm
            Q = cp.Parameter((self.s, self.s), value=self.gamma**2*np.eye(self.s)) 
            S = cp.Parameter((self.s, self.s), value=np.zeros((self.s, self.s)))
            R = cp.Parameter((self.s, self.s), value=-np.eye(self.s)) 

        return Q, S, R
    
    def sdp_solve(self):
        # CVXPY variables
        p = cp.Variable(self.n) # p1, p2, ..., pn

        # Q, S, R matrices
        Q, S, R = self.construct_QRS()
        print(Q)

        # Define bigger [Q, S; S' R] matrix
        block_Q = cp.vstack([cp.hstack([cp.multiply(p[i], Q) if i == j else cp.Constant(np.zeros((self.s, self.s))) for j in range(self.n)]) for i in range(self.n)])
        block_S = cp.vstack([cp.hstack([cp.multiply(p[i], S) if i == j else cp.Constant(np.zeros((self.s, self.s))) for j in range(self.n)]) for i in range(self.n)])
        block_ST = cp.vstack([cp.hstack([cp.multiply(p[i], S.T) if i == j else cp.Constant(np.zeros((self.s, self.s))) for j in range(self.n)]) for i in range(self.n)])
        block_R = cp.vstack([cp.hstack([cp.multiply(p[i], R) if i == j else cp.Constant(np.zeros((self.s, self.s))) for j in range(self.n)]) for i in range(self.n)])

        QS = cp.hstack((block_Q, block_S)) # [Q, S]
        STR = cp.hstack((block_ST, block_R)) # [S', R]
        QSSR = cp.vstack((QS, STR)) # [Q, S; S', R]

        # Define MI matrix
        I = np.eye(self.M.shape[0])
        MI = cp.vstack((self.M, I))

        # DSP matrix
        stability_matrix = cp.matmul(MI.T, cp.matmul(QSSR, MI)) # MI.T * [Q, S; S', R] * MI

        # DSP problem
        constraints = [-stability_matrix >> 0]
        epsilon = 1e-6  # Small positive number
        constraints.append(p >= epsilon)  # Approximate strict positivity

        # Objective
        objective = cp.Minimize(0)

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()

        return p.value
    
def get_W_from_A(A):
    W = A.T # Transpose
    InD = np.sum(A, axis = 0)
    for i in range(A.shape[0]): # Diagonal elements = -in-degree
        W[i, i] = -InD[i]
    return W
    
if __name__ == "__main__":

    # Example

    num_nodes = 3
    state_dim = 1 # Only output state dimension

    A = np.array([[0, 1, 1],
                [1, 0, 0],
                [0, 1, 0]]) # Adjacency matrix
    
    stability = Stability(M = get_W_from_A(A), s = state_dim, dissipativity = 'L2', gamma = 0.99)
    p = stability.sdp_solve()
    print(p)