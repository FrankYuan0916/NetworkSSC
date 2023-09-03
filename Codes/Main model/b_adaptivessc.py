from a_preData import *

class ADAPTIVESSC:

    def __init__(self, X, label):
        self.X = X
        self.label = label
        self.max_iter = 200
        self.gamma = 1   
        self.mu = 50

    def solve_l1(self, b, lamda):
        max_term = abs(b) - lamda
        x = np.sign(b) * np.maximum(max_term, np.zeros(max_term.shape))
        return x   

    def adaptiveSSC(self, X, lamda, W):

        C = np.zeros((X.shape[1], X.shape[1]))
        J = np.zeros((X.shape[1], X.shape[1]))
        Y = np.zeros((X.shape[1], X.shape[1]))
        tmp1 = np.transpose(X).dot(X) + 1/self.gamma * np.identity(X.shape[1])
        tmp2 = np.transpose(X).dot(X)

        for i in range(self.max_iter):

            C_prev = C
            J_prev = J
            
            C = np.linalg.inv(tmp1).dot(tmp2 + 1/self.gamma * (J - Y))
            C = C - np.diag(np.diag(C))
            
            V = C + Y
            W_reci = np.reciprocal(W)
            J = self.solve_l1(V, lamda/self.gamma * W_reci)
            J = J - np.diag(np.diag(J))
            # J[J < 0] = 0
            
            Y = Y + 1/self.gamma * (C - J)
            
            if np.max(abs(C - J)) < 0.0001:
                break
            
            if i % 10 == 0 and i != 0:
                gamma_primal = C - J
                gamma_primal = np.linalg.norm(gamma_primal, 2)
                gamma_dual = (J - J_prev) / self.gamma
                gamma_dual = np.linalg.norm(gamma_dual, 2)
                if gamma_primal > self.mu * gamma_dual:
                    self.gamma = self.gamma / 2
                elif gamma_dual > self.mu * gamma_primal:
                    self.gamma = 2 * self.gamma

        S = np.transpose(C) + C
        return S
